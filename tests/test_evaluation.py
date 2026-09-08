from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ndr.evaluation import (
    calibrate_threshold,
    constrained_operating_point,
    empirical_tail_strength,
    fuse,
    recall_floor,
    validate_score_cache,
)


def test_benign_threshold_and_tail_fusion_are_tie_safe() -> None:
    calibration = np.array([1.0, 2.0, 2.0, 4.0])
    metadata = calibrate_threshold(calibration, 0.75)
    assert metadata["labels_used"] is False
    tail = empirical_tail_strength(np.array([0.0, 2.0, 5.0]), calibration)
    assert np.isfinite(tail).all()
    np.testing.assert_array_equal(fuse(tail, tail + 1, 1.0), tail)
    labels = np.array([False, False, True, True])
    point = recall_floor(np.array([0.1, 0.2, 0.2, 0.3]), labels, 2)
    assert point["true_positives"] == 2


def test_precision_and_recall_constraints_choose_opposite_objectives() -> None:
    scores = np.array([0.1, 0.8, 0.3, 0.9, 0.7, 0.2])
    labels = np.array([False, False, False, True, True, True])
    precision = constrained_operating_point(scores, labels, min_precision=0.6)
    recall = constrained_operating_point(scores, labels, min_recall=2 / 3)
    assert precision["precision"] >= 0.6
    assert recall["recall"] >= 2 / 3
    assert precision["labels_used"] and recall["labels_used"]
    with pytest.raises(ValueError, match="exactly one"):
        constrained_operating_point(scores, labels)


def test_score_cache_rejects_duplicate_identities() -> None:
    frame = pd.DataFrame({
        "uid": ["a", "a"], "ts": [1.0, 1.0], "period": ["development"] * 2,
        "is_attack": [False, True], "ocsvm_score": [0.1, 0.2], "ae_score": [0.1, 0.2],
        "fused_score": [float("nan"), float("nan")],
    })
    with pytest.raises(ValueError, match="duplicate"):
        validate_score_cache(frame)
