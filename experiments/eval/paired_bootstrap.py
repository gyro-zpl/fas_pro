"""Paired bootstrap: LLM vs GBM vs base on the same in-distribution rows.

Outputs:
    experiments/results/paper_tables/paired_bootstrap.json
    experiments/results/paper_tables/paired_bootstrap.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics

IN_DIST = [2016, 2018, 2019]
KEY = ["respondent_id", "year", "candidate"]


def _load(stem: str) -> pd.DataFrame:
    df = pd.read_json(config.PREDS / f"{stem}.predictions.jsonl", lines=True)
    df = df[df["year"].isin(IN_DIST)].copy()
    return df[KEY + ["true_score", "pred_float"]]


def _paired(a: pd.DataFrame, b: pd.DataFrame, n_boot: int = 1000, seed: int = 42) -> dict:
    b = b.drop(columns=["true_score"])
    df = a.merge(b, on=KEY, how="inner", suffixes=("_a", "_b"))
    df = df.rename(columns={"pred_float_a": "a", "pred_float_b": "b"})
    df = df.dropna(subset=["true_score", "a", "b"]).reset_index(drop=True)

    y = df["true_score"].to_numpy(float)
    pa = df["a"].to_numpy(float)
    pb = df["b"].to_numpy(float)
    err_a, err_b = np.abs(y - pa), np.abs(y - pb)

    # Precompute per-group correctness for the choice metric.
    correct_a, correct_b = [], []
    for _, g in df.groupby(["respondent_id", "year"]):
        if len(g) < 2:
            continue
        true_choice = g["true_score"].idxmax()
        correct_a.append(int(g["a"].idxmax() == true_choice))
        correct_b.append(int(g["b"].idxmax() == true_choice))
    correct_a = np.asarray(correct_a, float)
    correct_b = np.asarray(correct_b, float)

    rng = np.random.default_rng(seed)
    n = len(df)
    n_groups = len(correct_a)
    d_mae, d_corr, d_choice = [], [], []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        d_mae.append(float(err_a[idx].mean() - err_b[idx].mean()))
        yy, aa, bb = y[idx], pa[idx], pb[idx]
        if aa.std() > 1e-9 and bb.std() > 1e-9:
            d_corr.append(float(np.corrcoef(yy, aa)[0, 1] - np.corrcoef(yy, bb)[0, 1]))
        gidx = rng.integers(0, n_groups, n_groups)
        d_choice.append(float(correct_a[gidx].mean() - correct_b[gidx].mean()))

    def summarize(vals):
        vals = np.array([v for v in vals if v == v])
        lo, hi = np.percentile(vals, [2.5, 97.5])
        return {
            "mean": float(vals.mean()),
            "lo": float(lo),
            "hi": float(hi),
            "p_sign": float(min((vals <= 0).mean(), (vals >= 0).mean()) * 2),
        }

    return {"n": int(n), "delta_mae": summarize(d_mae),
            "delta_corr": summarize(d_corr), "delta_choice": summarize(d_choice)}


def main() -> None:
    llm = {s: _load(f"qwen3_t_seed{s}_test_r_generate") for s in [42, 43, 44]}
    gbm = _load("tabular_histgb_test_r")
    base = _load("qwen3_base_test_r_generate")

    table = {}
    for s, df in llm.items():
        table[f"llm_seed{s}_minus_gbm"] = _paired(df, gbm)
    table["base_minus_gbm"] = _paired(base, gbm)
    table["llm_seed42_minus_base"] = _paired(llm[42], base)

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "paired_bootstrap.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    lines = ["# Paired bootstrap (in-distribution 2016-2019)", "",
             "Positive delta means the first model is worse (MAE) / better (corr, choice).", "",
             "| comparison | n | ΔMAE [95% CI] | Δcorr [95% CI] | Δchoice [95% CI] |",
             "|---|---|---|---|---|"]
    for name, r in table.items():
        lines.append(
            f"| {name} | {r['n']} | {r['delta_mae']['mean']:+.2f} "
            f"[{r['delta_mae']['lo']:+.2f}, {r['delta_mae']['hi']:+.2f}] | "
            f"{r['delta_corr']['mean']:+.3f} [{r['delta_corr']['lo']:+.3f}, {r['delta_corr']['hi']:+.3f}] | "
            f"{r['delta_choice']['mean']:+.3f} [{r['delta_choice']['lo']:+.3f}, {r['delta_choice']['hi']:+.3f}] |"
        )
    (out_dir / "paired_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
