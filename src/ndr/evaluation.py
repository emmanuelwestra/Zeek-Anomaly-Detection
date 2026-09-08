from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def calibrate_threshold(scores: np.ndarray, quantile: float) -> dict[str, Any]:
    values = _finite(scores)
    if not 0 < quantile < 1:
        raise ValueError("quantile must be between 0 and 1")
    if not len(values):
        raise ValueError("Cannot calibrate on no scores")
    return {
        "mode": "benign_validation_quantile",
        "quantile": float(quantile),
        "threshold": float(np.quantile(values, quantile)),
        "validation_rows": int(len(values)),
        "minimum_score": float(values.min()),
        "median_score": float(np.median(values)),
        "maximum_score": float(values.max()),
        "labels_used": False,
    }


def flow_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    y = np.asarray(labels, dtype=bool)
    values = _finite(scores)
    if y.shape != values.shape:
        raise ValueError("Labels and scores must be row-aligned")
    predicted = values > threshold
    tp = int(np.sum(y & predicted))
    tn = int(np.sum(~y & ~predicted))
    fp = int(np.sum(~y & predicted))
    fn = int(np.sum(y & ~predicted))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    accuracy = (tp + tn) / len(y) if len(y) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    denominator = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {
        "rows": int(len(y)),
        "threshold": float(threshold),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": float((tp * tn - fp * fn) / np.sqrt(denominator)) if denominator else 0.0,
        "auroc": float(roc_auc_score(y, values)) if np.unique(y).size == 2 else None,
        "auprc": float(average_precision_score(y, values)) if np.unique(y).size == 2 else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "false_negative_rate": fn / (fn + tp) if fn + tp else None,
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "alerts": int(predicted.sum()),
        "alerts_per_10000_events": float(predicted.mean() * 10_000) if len(y) else 0.0,
    }


def evaluate_frame(frame: pd.DataFrame, score_column: str, threshold: float) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {"ts", "is_attack", score_column}
    missing = required - set(frame)
    if missing:
        raise KeyError(f"Evaluation frame is missing {sorted(missing)}")
    summary = flow_metrics(frame["is_attack"].to_numpy(), frame[score_column].to_numpy(), threshold)
    working = frame[["ts", "is_attack", score_column]].copy()
    working["hour"] = pd.to_datetime(working["ts"], unit="s", utc=True).dt.floor("h")
    working["alert"] = working[score_column].gt(threshold)
    windows = working.groupby("hour", sort=True).agg(
        rows=("is_attack", "size"),
        attacks=("is_attack", "sum"),
        alerts=("alert", "sum"),
        has_attack=("is_attack", "any"),
        is_alert_window=("alert", "any"),
    ).reset_index()
    tp = int((windows.has_attack & windows.is_alert_window).sum())
    fp = int((~windows.has_attack & windows.is_alert_window).sum())
    fn = int((windows.has_attack & ~windows.is_alert_window).sum())
    tn = int((~windows.has_attack & ~windows.is_alert_window).sum())
    summary.update({
        "windows": int(len(windows)),
        "window_true_positives": tp,
        "window_true_negatives": tn,
        "window_false_positives": fp,
        "window_false_negatives": fn,
        "window_precision": tp / (tp + fp) if tp + fp else 0.0,
        "window_recall": tp / (tp + fn) if tp + fn else 0.0,
    })
    return summary, windows


def empirical_tail_strength(scores: np.ndarray, benign_scores: np.ndarray) -> np.ndarray:
    values = _finite(scores)
    calibration = np.sort(_finite(benign_scores))
    if not len(calibration):
        raise ValueError("At least one benign calibration score is required")
    tail_count = len(calibration) - np.searchsorted(calibration, values, side="left")
    return -np.log((1 + tail_count) / (len(calibration) + 1))


def fuse(ocsvm_tail: np.ndarray, ae_tail: np.ndarray, alpha: float) -> np.ndarray:
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")
    left, right = _finite(ocsvm_tail), _finite(ae_tail)
    if left.shape != right.shape:
        raise ValueError("Fusion scores must be row-aligned")
    return alpha * left + (1 - alpha) * right


def recall_floor(scores: np.ndarray, labels: np.ndarray, true_positive_floor: int) -> dict[str, Any]:
    values, y = _finite(scores), np.asarray(labels, dtype=bool)
    attacks = values[y]
    if not 0 <= true_positive_floor <= len(attacks):
        raise ValueError("true-positive floor is outside the attack count")
    if true_positive_floor == 0:
        threshold = float(values.max())
    else:
        cutoff = float(np.partition(attacks, len(attacks) - true_positive_floor)[len(attacks) - true_positive_floor])
        threshold = cutoff
        if int(np.sum(attacks > threshold)) < true_positive_floor:
            threshold = float(np.nextafter(cutoff, -np.inf))
    result = flow_metrics(y, values, threshold)
    result["mode"] = "retrospective_recall_floor"
    result["labels_used"] = True
    return result


def constrained_operating_point(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    min_precision: float | None = None,
    min_recall: float | None = None,
) -> dict[str, Any]:
    """Find the best retrospective threshold under exactly one constraint."""

    if (min_precision is None) == (min_recall is None):
        raise ValueError("Specify exactly one of min_precision or min_recall")
    constraint = min_precision if min_precision is not None else min_recall
    if constraint is None or not 0 <= constraint <= 1:
        raise ValueError("Constraint must be between 0 and 1")
    values, y = _finite(scores), np.asarray(labels, dtype=bool)
    if values.shape != y.shape or not len(values):
        raise ValueError("Scores and labels must be non-empty and row-aligned")
    order = np.argsort(-values, kind="mergesort")
    ordered_scores, ordered_labels = values[order], y[order]
    end = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    indices = np.flatnonzero(end)
    tp = np.cumsum(ordered_labels)[indices]
    fp = (indices + 1) - tp
    attack_count = int(y.sum())
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / attack_count if attack_count else np.zeros_like(tp, dtype=float)
    eligible = precision >= constraint if min_precision is not None else recall >= constraint
    if not np.any(eligible):
        name = "precision" if min_precision is not None else "recall"
        raise ValueError(f"No threshold satisfies minimum {name} {constraint}")
    candidates = np.flatnonzero(eligible)
    if min_precision is not None:
        best = candidates[np.lexsort((-precision[candidates], -recall[candidates]))[0]]
        objective = "maximum_recall_at_minimum_precision"
    else:
        best = candidates[np.lexsort((-recall[candidates], -precision[candidates]))[0]]
        objective = "maximum_precision_at_minimum_recall"
    threshold = float(np.nextafter(ordered_scores[indices[best]], -np.inf))
    result = flow_metrics(y, values, threshold)
    result.update({
        "mode": "retrospective_constraint",
        "objective": objective,
        "minimum_precision": min_precision,
        "minimum_recall": min_recall,
        "labels_used": True,
        "prevalence": float(y.mean()),
    })
    return result


def validate_score_cache(frame: pd.DataFrame) -> None:
    required = {
        "uid", "ts", "period", "is_attack", "ocsvm_score", "ae_score", "fused_score"
    }
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Score cache is missing {sorted(missing)}")
    if frame[["uid", "ts"]].isna().any().any():
        raise ValueError("Score cache identities cannot be missing")
    if frame.duplicated(["uid", "ts"]).any():
        raise ValueError("Score cache contains duplicate (uid, ts) identities")
    if not np.isfinite(frame[["ocsvm_score", "ae_score"]].to_numpy(dtype=float)).all():
        raise ValueError("Score cache contains non-finite model scores")
    fused = frame["fused_score"].to_numpy(dtype=float)
    if not (np.isnan(fused).all() or np.isfinite(fused).all()):
        raise ValueError("Fused scores must be either entirely pending or entirely finite")


def load_score_cache(path: str | Path, period: str | None = None) -> pd.DataFrame:
    if Path(path).is_dir():
        pattern = f"{period}-*.parquet" if period else "*.parquet"
        files = sorted(Path(path).glob(pattern))
    else:
        files = [Path(path)]
    if not files or not all(file.is_file() for file in files):
        raise FileNotFoundError(f"No score cache found at {path}")
    frame = pd.concat([pd.read_parquet(file) for file in files], ignore_index=True)
    validate_score_cache(frame)
    if period:
        frame = frame.loc[frame.period.eq(period)].copy()
        if frame.empty:
            raise ValueError(f"Score cache has no {period} rows")
    return frame


def _finite(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=float)
    if result.ndim != 1 or not np.isfinite(result).all():
        raise ValueError("Scores must be a finite one-dimensional array")
    return result
