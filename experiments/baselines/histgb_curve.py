"""Dump the HistGB train/validation loss curve for the PAS baseline.

Fits the same model as `experiments/baselines/tabular.py` (identical features,
split and hyperparameters) and writes the per-iteration loss, the early-stopping
point and an optional PNG.

Outputs:
    experiments/results/paper_tables/histgb_curve.json
    experiments/results/paper_tables/histgb_curve.md
    experiments/results/paper_tables/histgb_curve.png   (if matplotlib exists)

Usage:
    python -m experiments.baselines.histgb_curve --train-split train_r
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments import config
from experiments.baselines.tabular import CAT_COLS, _design, _load_split


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="train_r")
    args = ap.parse_args()

    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean = clean.set_index("orig_row_id", drop=False)
    train = _load_split(config.V3_SPLITS / f"{args.train_split}.jsonl")
    rows = clean.loc[train["orig_row_id"]].reset_index(drop=True)

    X = _design(rows, CAT_COLS)
    y = train["true_score"].values

    model = HistGradientBoostingRegressor(
        max_iter=60, learning_rate=0.08, max_bins=64,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=42,
    )
    model.fit(X, y)

    train_curve = [float(v) for v in model.train_score_]
    val_curve = [float(v) for v in model.validation_score_]

    out = {
        "train_split": args.train_split,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "max_iter": 60,
        "n_iter_used": int(model.n_iter_),
        "loss": "squared_error (sklearn default)",
        "train_score": train_curve,
        "validation_score": val_curve,
        "final_train": train_curve[-1] if train_curve else None,
        "final_validation": val_curve[-1] if val_curve else None,
    }

    out_dir = Path(config.RESULTS) / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "histgb_curve.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    lines = [
        "# HistGB training curve",
        "",
        f"- split: `{args.train_split}`, rows {out['n_rows']:,}, features {out['n_features']}",
        f"- early stopping stopped at **{out['n_iter_used']} / 60** trees",
        f"- final train loss {out['final_train']:.2f}, final validation loss {out['final_validation']:.2f}",
        "",
        "| tree | train | validation |",
        "|---:|---:|---:|",
    ]
    for i, (tr, va) in enumerate(zip(train_curve, val_curve), 1):
        lines.append(f"| {i} | {tr:.2f} | {va:.2f} |")
    (out_dir / "histgb_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(range(1, len(train_curve) + 1), train_curve, label="train")
        ax.plot(range(1, len(val_curve) + 1), val_curve, label="validation (10%)")
        ax.axvline(out["n_iter_used"], ls=":", lw=1, color="grey")
        ax.set_xlabel("trees"); ax.set_ylabel("squared error")
        ax.set_title(f"HistGB training curve ({out['n_iter_used']}/60 trees)")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / "histgb_curve.png", dpi=200)
    except Exception as exc:  # matplotlib optional
        print(f"[warn] plot skipped: {exc}")

    print(json.dumps({k: out[k] for k in
                      ("n_rows", "n_features", "n_iter_used", "final_train", "final_validation")}))


if __name__ == "__main__":
    main()
