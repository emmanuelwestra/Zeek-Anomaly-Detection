from __future__ import annotations

import itertools
import platform
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import sklearn
import torch

from ndr.config import load_grid, write_yaml
from ndr.data import Corpus
from ndr.evaluation import (
    calibrate_threshold,
    constrained_operating_point,
    empirical_tail_strength,
    evaluate_frame,
    flow_metrics,
    fuse,
    load_score_cache,
    recall_floor,
    validate_score_cache,
)
from ndr.features import FlowPreprocessor
from ndr.models import OCSVM, MixedAutoencoder, combine_ae_components


def paths(config: dict[str, Any]) -> dict[str, Path]:
    artifacts = config["artifacts"]
    return {
        "ocsvm_model": Path(artifacts["models_dir"]) / "ocsvm.joblib",
        "ae_model": Path(artifacts["models_dir"]) / "mixed_ae.pt",
        "ocsvm_preprocessor": Path(artifacts["preprocessors_dir"]) / "ocsvm.joblib",
        "ae_preprocessor": Path(artifacts["preprocessors_dir"]) / "mixed_ae.joblib",
        "ocsvm_threshold": Path(artifacts["thresholds_dir"]) / "ocsvm.yaml",
        "ae_threshold": Path(artifacts["thresholds_dir"]) / "mixed_ae.yaml",
        "fusion": Path(artifacts["thresholds_dir"]) / "fusion.yaml",
        "cache": Path(artifacts["score_cache_dir"]),
        "reports": Path(artifacts["reports_dir"]),
    }


def build_ocsvm(model_config: dict[str, Any]) -> OCSVM:
    return OCSVM(**{key: model_config[key] for key in (
        "kernel", "gamma", "nu", "cache_size", "tolerance", "shrinking",
        "scoring_threads", "random_state",
    ) if key in model_config})


def build_ae(model_config: dict[str, Any], schema: dict[str, Any]) -> MixedAutoencoder:
    keys = (
        "hidden_dims", "latent_dim", "activation", "normalization", "dropout",
        "categorical_loss", "training_numeric_weight", "scoring_numeric_weight", "epochs",
        "batch_size", "learning_rate", "weight_decay", "patience",
        "internal_validation_fraction", "random_state", "device",
    )
    return MixedAutoencoder(schema, **{key: model_config[key] for key in keys if key in model_config})


def train(config: dict[str, Any], model: str) -> list[dict[str, Any]]:
    if model not in {"ocsvm", "ae", "all"}:
        raise ValueError("model must be ocsvm, ae, or all")
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    train_frame, validation_frame = corpus.train(), corpus.validation()
    if train_frame["is_attack"].any() or validation_frame["is_attack"].any():
        raise ValueError("Training and validation splits must be attack-free")
    selected = ("ocsvm", "ae") if model == "all" else (model,)
    output = paths(config)
    results = []
    for name in selected:
        preprocessor = FlowPreprocessor.for_model(name, config["features"])
        x_train = preprocessor.fit_transform(train_frame)
        x_validation = preprocessor.transform(validation_frame)
        expected = config["features"].get(f"{name}_expected_width")
        if expected is not None and x_train.shape[1] != int(expected):
            raise ValueError(f"{name} feature contract expected {expected} columns, found {x_train.shape[1]}")
        started = perf_counter()
        if name == "ocsvm":
            detector = build_ocsvm(config["models"][name]).fit(x_train)
        else:
            detector = build_ae(config["models"][name], preprocessor.reconstruction_schema()).fit(x_train)
        seconds = perf_counter() - started
        validation_scores = detector.score_samples(x_validation)
        threshold = calibrate_threshold(validation_scores, float(config["thresholds"][f"{name}_quantile"]))
        threshold.update({"model": name, "training_rows": len(x_train), "feature_width": x_train.shape[1]})
        detector.save(output[f"{name}_model"])
        preprocessor.save(output[f"{name}_preprocessor"])
        write_yaml(threshold, output[f"{name}_threshold"])
        if name == "ae":
            pd.DataFrame(detector.history_).to_csv(output["reports"] / "ae_training_history.csv", index=False)
        results.append({
            "model": name,
            "training_rows": len(x_train),
            "feature_width": x_train.shape[1],
            "training_seconds": seconds,
            "training_events_per_second": len(x_train) / seconds,
            "support_vectors": detector.support_vectors if name == "ocsvm" else None,
            "threshold": threshold["threshold"],
            "threshold_mode": threshold["mode"],
        })
    pd.DataFrame(results).to_csv(output["reports"] / "training_summary.csv", index=False)
    return results


def score(config: dict[str, Any], period: str) -> Path:
    """Score both models once and persist an identity-aligned cache."""

    output = paths(config)
    output["cache"].mkdir(parents=True, exist_ok=True)
    for stale in output["cache"].glob(f"{period}-*.parquet"):
        stale.unlink()
    oc_pre = FlowPreprocessor.load(output["ocsvm_preprocessor"])
    ae_pre = FlowPreprocessor.load(output["ae_preprocessor"])
    ocsvm = OCSVM.load(output["ocsvm_model"])
    ae = MixedAutoencoder.load(output["ae_model"])
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    total_rows = 0
    part_count = 0
    for part_index, frame in enumerate(corpus.iter_period(period, config["splits"]["holdout_start"])):
        if frame.duplicated(["uid", "ts"]).any():
            raise ValueError("Prepared partition contains duplicate identities")
        cached = frame[["uid", "ts", "is_attack", "source_dataset", "capture_week_start"]].copy()
        cached["period"] = period
        cached["ocsvm_score"] = ocsvm.score_samples(oc_pre.transform(frame))
        cached["ae_score"] = ae.score_samples(ae_pre.transform(frame))
        cached["fused_score"] = np.nan
        validate_score_cache(cached)
        target = output["cache"] / f"{period}-{part_index:03d}.parquet"
        cached.to_parquet(target, index=False)
        total_rows += len(cached)
        part_count += 1
    if not part_count:
        raise ValueError(f"No prepared rows found for {period}")
    manifest = {
        "period": period,
        "rows": total_rows,
        "parts": part_count,
        "identity": ["uid", "ts"],
        "scores": ["ocsvm_score", "ae_score", "fused_score"],
    }
    write_yaml(manifest, output["cache"] / f"{period}-manifest.yaml")
    return output["cache"]


def validation_scores(config: dict[str, Any]) -> pd.DataFrame:
    output = paths(config)
    target = output["cache"] / "validation.parquet"
    if target.is_file():
        frame = pd.read_parquet(target)
        validate_score_cache(frame)
        return frame
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    frame = corpus.validation()
    oc_pre = FlowPreprocessor.load(output["ocsvm_preprocessor"])
    ae_pre = FlowPreprocessor.load(output["ae_preprocessor"])
    result = frame[["uid", "ts", "is_attack"]].copy()
    result["period"] = "validation"
    result["ocsvm_score"] = OCSVM.load(output["ocsvm_model"]).score_samples(oc_pre.transform(frame))
    result["ae_score"] = MixedAutoencoder.load(output["ae_model"]).score_samples(ae_pre.transform(frame))
    result["fused_score"] = np.nan
    validate_score_cache(result)
    target.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(target, index=False)
    return result


def evaluate(config: dict[str, Any], model: str, period: str) -> dict[str, Any]:
    if model not in {"ocsvm", "ae", "fusion"}:
        raise ValueError("model must be ocsvm, ae, or fusion")
    output = paths(config)
    try:
        cache = load_score_cache(output["cache"], period)
    except (FileNotFoundError, ValueError):
        score(config, period)
        cache = load_score_cache(output["cache"], period)
    if model == "fusion":
        selected = _load_yaml(output["fusion"])
        validation = validation_scores(config)
        cache["fusion_score"] = fuse(
            empirical_tail_strength(cache.ocsvm_score, validation.ocsvm_score),
            empirical_tail_strength(cache.ae_score, validation.ae_score),
            float(selected["alpha"]),
        )
        cache["fused_score"] = cache["fusion_score"]
        _persist_fused_scores(output["cache"], period, cache["fused_score"].to_numpy())
        score_column, threshold = "fusion_score", float(selected["threshold"])
        threshold_mode = selected["mode"]
    else:
        score_column = f"{model}_score"
        threshold_data = _load_yaml(output[f"{model}_threshold"])
        threshold, threshold_mode = float(threshold_data["threshold"]), threshold_data["mode"]
    summary, windows = evaluate_frame(cache, score_column, threshold)
    summary.update({"model": model, "period": period, "threshold_mode": threshold_mode})
    if model == "fusion":
        summary["alpha"] = float(selected["alpha"])
    write_yaml(summary, output["reports"] / f"{model}_{period}_metrics.yaml")
    pd.DataFrame([summary]).to_csv(output["reports"] / f"{model}_{period}_metrics.csv", index=False)
    windows.to_csv(output["reports"] / f"{model}_{period}_window_metrics.csv", index=False)
    return summary


def search_ocsvm(config: dict[str, Any], grid_path: str | Path) -> dict[str, Any]:
    grid = load_grid(grid_path, "search")
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    train_frame, validation = corpus.train(), corpus.validation()
    development = pd.concat(list(corpus.iter_period("development", config["splits"]["holdout_start"])), ignore_index=True)
    preprocessor = FlowPreprocessor.for_model("ocsvm", config["features"])
    x_train = preprocessor.fit_transform(train_frame)
    x_validation = preprocessor.transform(validation)
    x_development = preprocessor.transform(development)
    rows = []
    for gamma, nu in itertools.product(grid["gamma"], grid["nu"]):
        model_config = {**config["models"]["ocsvm"], "gamma": gamma, "nu": nu}
        detector = build_ocsvm(model_config).fit(x_train)
        validation_score = detector.score_samples(x_validation)
        development_score = detector.score_samples(x_development)
        for quantile in grid["quantiles"]:
            threshold = float(np.quantile(validation_score, quantile))
            metrics = flow_metrics(development.is_attack.to_numpy(), development_score, threshold)
            rows.append({
                "run_name": f"g{gamma:g}_nu{nu:g}_q{quantile:g}",
                "gamma": gamma,
                "nu": nu,
                "quantile": quantile,
                **metrics,
            })
    results = pd.DataFrame(rows)
    results["eligible"] = results.recall.ge(float(grid["target_recall"]))
    eligible = results.loc[results.eligible]
    if eligible.empty:
        raise ValueError("No OC-SVM candidate reaches the development recall target")
    selected = eligible.sort_values(
        ["f1", "mcc", "auprc", "precision", "false_positive_rate"],
        ascending=[False, False, False, False, True], kind="mergesort"
    ).iloc[0]
    results["selected"] = False
    results.loc[selected.name, "selected"] = True
    output = paths(config)["reports"]
    results.to_csv(output / "ocsvm_search.csv", index=False)
    selection = _selection_record(selected, "development", [42])
    write_yaml(selection, output / "ocsvm_search_selected.yaml")
    return selection


def search_ae(config: dict[str, Any], grid_path: str | Path) -> dict[str, Any]:
    grid = load_grid(grid_path, "search")
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    train_frame = corpus.train()
    development = pd.concat(list(corpus.iter_period("development", config["splits"]["holdout_start"])), ignore_index=True)
    preprocessor = FlowPreprocessor.for_model("ae", config["features"])
    x_train = preprocessor.fit_transform(train_frame)
    x_development = preprocessor.transform(development)
    attack_count = int(development.is_attack.sum())
    true_positive_floor = int(np.ceil(float(grid["target_recall"]) * attack_count))
    rows = []
    checkpoint_dir = Path(config["artifacts"]["models_dir"]) / "search_checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    for training_weight, seed in itertools.product(grid["training_numeric_weight"], grid["seeds"]):
        model_config = {
            **config["models"]["ae"],
            "training_numeric_weight": training_weight,
            "scoring_numeric_weight": training_weight,
            "random_state": seed,
        }
        detector = build_ae(model_config, preprocessor.reconstruction_schema()).fit(x_train)
        components = detector.components(x_development)
        detector.save(checkpoint_dir / f"train{training_weight:.2f}_seed{seed}.pt")
        for scoring_weight in grid["scoring_numeric_weight"]:
            candidate_scores = combine_ae_components(components["numeric"], components["categorical"], scoring_weight)
            metrics = recall_floor(candidate_scores, development.is_attack.to_numpy(), true_positive_floor)
            rows.append({
                "run_name": f"train{training_weight:g}_score{scoring_weight:g}_seed{seed}",
                "training_numeric_weight": training_weight,
                "scoring_numeric_weight": scoring_weight,
                "seed": seed,
                "eligible": metrics["recall"] >= float(grid["target_recall"]),
                **metrics,
            })
    candidates = pd.DataFrame(rows)
    group_keys = ["training_numeric_weight", "scoring_numeric_weight"]
    summary = candidates.groupby(group_keys, as_index=False).agg(
        seeds=("seed", "nunique"), min_recall=("recall", "min"), mean_precision=("precision", "mean"),
        mean_f1=("f1", "mean"), mean_mcc=("mcc", "mean"), mean_auprc=("auprc", "mean"),
        mean_auroc=("auroc", "mean"), max_fpr=("false_positive_rate", "max"),
    )
    summary["eligible"] = summary.seeds.eq(len(grid["seeds"])) & summary.min_recall.ge(float(grid["target_recall"]))
    eligible = summary.loc[summary.eligible]
    if eligible.empty:
        raise ValueError("No AE candidate reaches the cross-seed development recall target")
    selected = eligible.sort_values(
        ["mean_f1", "mean_mcc", "mean_auprc", "mean_precision", "max_fpr"],
        ascending=[False, False, False, False, True], kind="mergesort"
    ).iloc[0]
    summary["selected"] = False
    summary.loc[selected.name, "selected"] = True
    output = paths(config)["reports"]
    candidates.to_csv(output / "ae_search_candidates.csv", index=False)
    summary.to_csv(output / "ae_search.csv", index=False)
    selection = {
        "selection_period": "development",
        "labels_used": True,
        "target_recall": float(grid["target_recall"]),
        "seeds": list(grid["seeds"]),
        **_native(selected.to_dict()),
    }
    write_yaml(selection, output / "ae_search_selected.yaml")
    return selection


def search_fusion(config: dict[str, Any], grid_path: str | Path) -> dict[str, Any]:
    grid = load_grid(grid_path, "fusion")
    output = paths(config)
    try:
        development = load_score_cache(output["cache"], "development")
    except (FileNotFoundError, ValueError):
        score(config, "development")
        development = load_score_cache(output["cache"], "development")
    validation = validation_scores(config)
    oc_tail = empirical_tail_strength(development.ocsvm_score, validation.ocsvm_score)
    ae_tail = empirical_tail_strength(development.ae_score, validation.ae_score)
    oc_threshold = float(_load_yaml(output["ocsvm_threshold"])["threshold"])
    reference = flow_metrics(development.is_attack.to_numpy(), development.ocsvm_score.to_numpy(), oc_threshold)
    alpha_values = np.arange(float(grid["alpha_start"]), float(grid["alpha_stop"]) + 1e-12, float(grid["alpha_step"]))
    rows = []
    for alpha in alpha_values:
        combined = fuse(oc_tail, ae_tail, float(alpha))
        metrics = recall_floor(combined, development.is_attack.to_numpy(), reference["true_positives"])
        rows.append({
            "run_name": f"alpha{alpha:.3f}",
            "alpha": float(alpha),
            "eligible": metrics["true_positives"] >= reference["true_positives"],
            **metrics,
        })
    results = pd.DataFrame(rows)
    selected = results.loc[results.eligible].sort_values(
        ["false_positives", "precision", "f1", "alpha"],
        ascending=[True, False, False, False], kind="mergesort"
    ).iloc[0]
    results["selected"] = False
    results.loc[selected.name, "selected"] = True
    results.to_csv(output["reports"] / "fusion_search.csv", index=False)
    selection = {
        "alpha": float(selected.alpha),
        "threshold": float(selected.threshold),
        "mode": "development_matched_ocsvm_recall",
        "selection_period": "development",
        "labels_used": True,
        "ocsvm_reference_threshold": oc_threshold,
        "ocsvm_reference_recall": reference["recall"],
        "ocsvm_reference_true_positives": reference["true_positives"],
        "selection_metrics": _native(selected.drop(labels=["selected"], errors="ignore").to_dict()),
    }
    write_yaml(selection, output["fusion"])
    write_yaml(selection, output["reports"] / "fusion_search_selected.yaml")
    _persist_fused_scores(
        output["cache"],
        "development",
        fuse(oc_tail, ae_tail, float(selected.alpha)),
    )
    return selection


def operating_point(
    config: dict[str, Any], model: str, period: str, min_precision: float | None, min_recall: float | None
) -> dict[str, Any]:
    output = paths(config)
    try:
        cache = load_score_cache(output["cache"], period)
    except (FileNotFoundError, ValueError):
        score(config, period)
        cache = load_score_cache(output["cache"], period)
    if model == "fusion":
        selection = _load_yaml(output["fusion"])
        validation = validation_scores(config)
        values = fuse(
            empirical_tail_strength(cache.ocsvm_score, validation.ocsvm_score),
            empirical_tail_strength(cache.ae_score, validation.ae_score),
            float(selection["alpha"]),
        )
    elif model in {"ocsvm", "ae"}:
        values = cache[f"{model}_score"].to_numpy()
    else:
        raise ValueError("model must be ocsvm, ae, or fusion")
    result = constrained_operating_point(
        values, cache.is_attack.to_numpy(), min_precision=min_precision, min_recall=min_recall
    )
    result.update({"model": model, "period": period, "warning": "Retrospective and prevalence-dependent; not deployable."})
    constraint = f"precision_{min_precision}" if min_precision is not None else f"recall_{min_recall}"
    write_yaml(result, output["reports"] / f"{model}_{period}_{constraint}_operating_point.yaml")
    return result


def benchmark(config: dict[str, Any], *, training_rows: list[int] | None = None) -> dict[str, Any]:
    output = paths(config)
    corpus = Corpus(Path(config["data"]["processed_dir"]))
    train_frame = corpus.train()
    sizes = training_rows or list(map(int, config["benchmark"]["training_rows"]))
    oc_pre = FlowPreprocessor.load(output["ocsvm_preprocessor"])
    ae_pre = FlowPreprocessor.load(output["ae_preprocessor"])
    x_oc = oc_pre.transform(train_frame)
    x_ae = ae_pre.transform(train_frame)
    rows = []
    for size in sizes:
        count = min(size, len(train_frame))
        for name, values in (("ocsvm", x_oc), ("ae", x_ae)):
            started = perf_counter()
            detector = (
                build_ocsvm(config["models"]["ocsvm"]).fit(values[:count])
                if name == "ocsvm"
                else build_ae(config["models"]["ae"], ae_pre.reconstruction_schema()).fit(values[:count])
            )
            seconds = perf_counter() - started
            rows.append(_benchmark_row("train", name, count, seconds, getattr(detector, "support_vectors", None)))
    pool = _inference_pool(
        corpus,
        max(config["benchmark"]["inference_rows"]),
        config["splits"]["holdout_start"],
    )
    score_matrices = {"ocsvm": oc_pre.transform(pool), "ae": ae_pre.transform(pool)}
    detectors = {"ocsvm": OCSVM.load(output["ocsvm_model"]), "ae": MixedAutoencoder.load(output["ae_model"])}
    warmup = int(config["benchmark"]["warmup_rows"])
    for name, detector in detectors.items():
        detector.score_samples(score_matrices[name][:warmup])
        for size in config["benchmark"]["inference_rows"]:
            count = min(int(size), len(pool))
            durations = []
            for _ in range(int(config["benchmark"]["inference_repetitions"])):
                started = perf_counter()
                detector.score_samples(score_matrices[name][:count])
                durations.append(perf_counter() - started)
            seconds = float(np.median(durations))
            rows.append(_benchmark_row("inference", name, count, seconds, getattr(detector, "support_vectors", None)))
    results = pd.DataFrame(rows)
    metadata = _runtime_metadata()
    for key, value in metadata.items():
        results[key] = value
    results.to_csv(output["reports"] / "benchmark.csv", index=False)
    fits = scaling_fits(results)
    summary = {"timing_scope": "model_only", "feature_transformation_timed": False, "fits": fits, **metadata}
    write_yaml(summary, output["reports"] / "benchmark_fits.yaml")
    return summary


def scaling_fits(results: pd.DataFrame) -> dict[str, Any]:
    training = results.loc[results.phase.eq("train")]
    fits: dict[str, Any] = {}
    oc = training.loc[training.model.eq("ocsvm")]
    ae = training.loc[training.model.eq("ae")]
    if len(oc) >= 2:
        exponent, log_scale = np.polyfit(np.log(oc.rows), np.log(oc.seconds), 1)
        fits["ocsvm_power"] = {"a": float(np.exp(log_scale)), "b": float(exponent), "formula": "seconds = a * rows ** b"}
    if len(ae) >= 2:
        slope, intercept = np.polyfit(ae.rows, ae.seconds, 1)
        fits["ae_linear"] = {"slope": float(slope), "intercept": float(intercept), "formula": "seconds = slope * rows + intercept"}
    return fits


def _inference_pool(corpus: Corpus, required: int, holdout_start: str) -> pd.DataFrame:
    frames, rows = [], 0
    for frame in corpus.iter_period("development", holdout_start):
        frames.append(frame)
        rows += len(frame)
        if rows >= required:
            break
    if not frames:
        raise ValueError("No development rows for inference benchmark")
    return pd.concat(frames, ignore_index=True).iloc[:required]


def _benchmark_row(phase: str, model: str, rows: int, seconds: float, support_vectors: int | None) -> dict[str, Any]:
    return {
        "phase": phase, "model": model, "rows": rows, "seconds": seconds,
        "events_per_second": rows / seconds if seconds else 0.0,
        "support_vectors": support_vectors,
    }


def _persist_fused_scores(cache_dir: Path, period: str, scores: np.ndarray) -> None:
    files = sorted(cache_dir.glob(f"{period}-*.parquet"))
    offset = 0
    for path in files:
        frame = pd.read_parquet(path)
        end = offset + len(frame)
        frame["fused_score"] = scores[offset:end]
        validate_score_cache(frame)
        frame.to_parquet(path, index=False)
        offset = end
    if offset != len(scores):
        raise ValueError("Fused scores are misaligned with the persisted score cache")


def _runtime_metadata() -> dict[str, Any]:
    return {
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "python_version": sys.version.split()[0],
        "scikit_learn_version": sklearn.__version__,
        "pytorch_version": torch.__version__,
    }


def _selection_record(selected: pd.Series, period: str, seeds: list[int]) -> dict[str, Any]:
    return {"selection_period": period, "labels_used": True, "seeds": seeds, **_native(selected.to_dict())}


def _native(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _native(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_native(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _load_yaml(path: Path) -> dict[str, Any]:
    import yaml

    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected mapping in {path}")
    return value
