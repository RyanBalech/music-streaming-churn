import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import f1_score
from churn.evaluation import (
    ModelConfig,
    bootstrap_intervals,
    choose_threshold,
    metrics,
    split_users,
    validate_predictions,
)


def test_user_splits_are_disjoint_and_reproducible():
    frame = pd.DataFrame({"id": np.arange(100), "target": [0, 1] * 50})
    parts = split_users(frame)
    assert [len(part) for part in parts] == [60, 20, 20]
    for a, b in ((0, 1), (1, 2), (0, 2)):
        assert set(parts[a].id).isdisjoint(parts[b].id)
    for actual, expected in zip(parts, split_users(frame)):
        pd.testing.assert_frame_equal(actual, expected)
    with pytest.raises(ValueError, match="one row per user"):
        split_users(pd.concat([frame, frame.iloc[:1]]))


def test_threshold_maximizes_validation_f1():
    y, p = np.array([0, 0, 1, 1, 0, 1]), np.array([0.1, 0.4, 0.3, 0.8, 0.2, 0.6])
    threshold = choose_threshold(y, p)
    assert f1_score(y, p >= threshold) == max(f1_score(y, p >= t) for t in p)


def test_imbalance_baseline_and_ranking_metrics():
    y = np.array([0] * 90 + [1] * 10)
    baseline = metrics(y, np.full(100, 0.1))
    assert baseline["accuracy"] == 0.9
    assert baseline["balanced_accuracy"] == 0.5
    assert baseline["average_precision"] == pytest.approx(0.1)
    assert baseline["roc_auc"] == 0.5
    perfect = metrics(y, y.astype(float))
    assert perfect["top_10_percent_lift"] == 10
    assert perfect["top_10_percent_recall"] == 1


def test_bootstrap_is_deterministic_and_handles_perfect_scores():
    y = np.array([0, 1] * 10)
    a = bootstrap_intervals(y, y.astype(float), repetitions=20)
    assert a == bootstrap_intervals(y, y.astype(float), repetitions=20)
    assert a["roc_auc_95"] == [1.0, 1.0]


@pytest.mark.parametrize(
    "y,p",
    [
        ([], []),
        ([[0, 1]], [[0.1, 0.2]]),
        ([0, 1], [0, np.nan]),
        ([0, 1], [0, 2]),
        ([2, 1], [0.1, 0.2]),
    ],
)
def test_invalid_predictions_are_rejected(y, p):
    with pytest.raises(ValueError):
        validate_predictions(y, p)


def test_invalid_configs_and_single_class_splits_fail():
    with pytest.raises(ValueError):
        ModelConfig(gb_weight=2)
    with pytest.raises(ValueError):
        split_users(pd.DataFrame({"id": np.arange(100), "target": [0] * 100}))
    with pytest.raises(ValueError):
        choose_threshold([0, 0], [0.1, 0.2])
