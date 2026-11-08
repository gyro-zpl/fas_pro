"""Plain-assert tests (pytest is not installed in the env)."""
from __future__ import annotations

import pandas as pd

from experiments.eval import metrics


def test_basic_metrics():
    y = [0, 50, 100]
    p = [10, 50, 90]
    assert abs(metrics.mae(y, p) - 20 / 3) < 1e-9
    assert abs(metrics.rmse(y, p) - (200 / 3) ** 0.5) < 1e-9
    assert metrics.relaxed_accuracy([50], [60]) == 1.0  # |diff| == 10
    assert metrics.relaxed_accuracy([50], [45]) == 1.0  # same 20-bin
    assert metrics.relaxed_accuracy([50], [61]) == 0.0  # different bin, diff 11
    assert metrics.relaxed_accuracy([50], [29]) == 0.0


def test_choice_accuracy():
    df = pd.DataFrame({
        "respondent_id": [1, 1, 2, 2],
        "year": [2020, 2020, 2020, 2020],
        "candidate": ["a", "b", "a", "b"],
        "true_score": [80, 20, 10, 90],
        "pred": [70, 30, 40, 60],
    })
    out = metrics.choice_accuracy(df)
    assert out["n_groups"] == 2
    assert out["accuracy"] == 1.0
    assert abs(out["chance"] - 0.5) < 1e-9


def test_choice_single_candidate_skipped():
    df = pd.DataFrame({
        "respondent_id": [1], "year": [2020],
        "candidate": ["a"], "true_score": [50], "pred": [50],
    })
    assert metrics.choice_accuracy(df)["n_groups"] == 0


def test_pairwise():
    df = pd.DataFrame({
        "respondent_id": [1, 1, 1],
        "year": [2020, 2020, 2020],
        "true_score": [90, 50, 10],
        "pred": [80, 60, 20],
    })
    out = metrics.pairwise_accuracy(df)
    assert out["n_pairs"] == 3
    assert out["accuracy"] == 1.0


def test_ece_perfect():
    y = [0, 20, 40, 60, 80, 100]
    assert metrics.ece(y, y) == 0.0


if __name__ == "__main__":
    for fn in [test_basic_metrics, test_choice_accuracy,
               test_choice_single_candidate_skipped, test_pairwise, test_ece_perfect]:
        fn()
        print("ok", fn.__name__)
