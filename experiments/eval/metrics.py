"""Evaluation metrics for PAS (rating + choice) and calibration."""
from __future__ import annotations

from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import stats


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def pearson(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if len(y_true) < 2:
        return float("nan")
    return float(stats.pearsonr(y_true, y_pred).statistic)


def spearman(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if len(y_true) < 2:
        return float("nan")
    return float(stats.spearmanr(y_true, y_pred).statistic)


def relaxed_accuracy(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    bin_size: int = 20,
    tol: int = 10,
) -> float:
    """Same 20-point bin OR absolute error <= tol (the legacy definition)."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if len(y_true) == 0:
        return float("nan")
    same_bin = (np.floor(y_true / bin_size) == np.floor(y_pred / bin_size))
    close = np.abs(y_true - y_pred) <= tol
    return float(np.mean(same_bin | close))


def choice_accuracy(
    df: pd.DataFrame,
    pred_col: str = "pred",
    true_col: str = "true_score",
    group_cols: tuple[str, ...] = ("respondent_id", "year"),
    min_candidates: int = 2,
) -> dict:
    """Rating-induced choice accuracy.

    For each (respondent, year) group, take the candidate with the highest
    predicted rating and compare it with the candidate with the highest true
    rating. Groups with fewer than `min_candidates` are skipped.
    """
    correct = 0
    groups = 0
    sizes = []
    per_year: dict[str, dict] = {}
    for keys, g in df.groupby(list(group_cols)):
        n = len(g)
        if n < min_candidates:
            continue
        true_choice = g.loc[g[true_col].idxmax()]
        pred_choice = g.loc[g[pred_col].idxmax()]
        hit = int(true_choice.name == pred_choice.name)
        correct += hit
        groups += 1
        sizes.append(n)
        year = str(keys[-1]) if isinstance(keys, tuple) else str(keys)
        rec = per_year.setdefault(year, {"correct": 0, "n": 0, "chance": 0.0})
        rec["correct"] += hit
        rec["n"] += 1
        rec["chance"] += 1.0 / n

    out = {
        "accuracy": correct / groups if groups else float("nan"),
        "n_groups": groups,
        "n_candidates_mean": float(np.mean(sizes)) if sizes else float("nan"),
        "chance": float(np.mean([1.0 / s for s in sizes])) if sizes else float("nan"),
        "by_year": {
            y: {
                "accuracy": v["correct"] / v["n"] if v["n"] else float("nan"),
                "n_groups": v["n"],
                "chance": v["chance"] / v["n"] if v["n"] else float("nan"),
            }
            for y, v in per_year.items()
        },
    }
    return out


def pairwise_accuracy(
    df: pd.DataFrame,
    pred_col: str = "pred",
    true_col: str = "true_score",
    group_cols: tuple[str, ...] = ("respondent_id", "year"),
) -> dict:
    """Pairwise ranking accuracy within each respondent-year group."""
    correct = 0.0
    total = 0
    for _, g in df.groupby(list(group_cols)):
        if len(g) < 2:
            continue
        for i, j in combinations(g.index, 2):
            ti, tj = g.at[i, true_col], g.at[j, true_col]
            if ti == tj:
                continue
            pi, pj = g.at[i, pred_col], g.at[j, pred_col]
            total += 1
            if pi == pj:
                correct += 0.5
            elif (ti - tj) * (pi - pj) > 0:
                correct += 1.0
    return {
        "accuracy": correct / total if total else float("nan"),
        "n_pairs": total,
    }


def ece(y_true: Iterable[float], y_pred: Iterable[float], n_bins: int = 10) -> float:
    """Expected calibration error for regression (bin by predicted value)."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    if len(y_true) == 0:
        return float("nan")
    bins = np.linspace(0, 100, n_bins + 1)
    idx = np.clip(np.digitize(y_pred, bins) - 1, 0, n_bins - 1)
    err = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        err += (m.sum() / len(y_pred)) * abs(y_true[m].mean() - y_pred[m].mean())
    return float(err)


def bootstrap_ci(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    metric_fn: Callable,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    n = len(y_true)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        vals.append(metric_fn(y_true[idx], y_pred[idx]))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def summarize(df: pd.DataFrame, pred_col: str = "pred", true_col: str = "true_score") -> dict:
    """Overall metrics plus arbitrary breakdowns the caller may add."""
    return {
        "n": int(len(df)),
        "mae": mae(df[true_col], df[pred_col]),
        "rmse": rmse(df[true_col], df[pred_col]),
        "pearson": pearson(df[true_col], df[pred_col]),
        "spearman": spearman(df[true_col], df[pred_col]),
        "relaxed_acc": relaxed_accuracy(df[true_col], df[pred_col]),
        "ece": ece(df[true_col], df[pred_col]),
        "choice": choice_accuracy(df, pred_col=pred_col, true_col=true_col),
        "pairwise": pairwise_accuracy(df, pred_col=pred_col, true_col=true_col),
    }
