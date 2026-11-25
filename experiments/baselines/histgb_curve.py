"""Dump the HistGB train/validation loss curve for the PAS baseline.

Fits the same model as `experiments/baselines/tabular.py` (identical features,
split and encoder) and writes the per-stage loss, the early-stopping point, the
test error of the fitted model, and an optional PNG.

`--max-iter` exists to test whether the 60-tree budget in the main table binds:
if validation loss is still falling when the budget runs out, the baseline is
not converged and its main-table MAE is an upper bound on the error a tuned
baseline could reach.

Outputs (suffix defaults to empty):
    experiments/results/paper_tables/histgb_curve<sfx>.json
    experiments/results/paper_tables/histgb_curve<sfx>.md
    experiments/results/paper_tables/histgb_curve<sfx>.png   (if matplotlib)

Usage:
    python -m experiments.baselines.histgb_curve --train-split train_r
    python -m experiments.baselines.histgb_curve --max-iter 400 --suffix _400
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from experiments import config
from experiments.baselines.tabular import CAT_COLS, _design, _load_split
from experiments.eval import metrics

MAIN_YEARS = [2016, 2018, 2019]


def _evaluate(model, clean: pd.DataFrame, split_name: str, columns) -> dict:
    """Score a fitted model on a split, plus the rows used by the main table."""
    path = config.V3_SPLITS / f"{split_name}.jsonl"
    if not path.exists():
        return {}
    split = _load_split(path)
    rows = clean.loc[split["orig_row_id"]].reset_index(drop=True)
    X = _design(rows, CAT_COLS).reindex(columns=columns, fill_value=0)
    frame = split.copy()
    frame["pred"] = np.clip(model.predict(X), 0, 100)
    frame["pred_float"] = frame["pred"]
    main = frame[frame["year"].isin(MAIN_YEARS)]
    out = {
        "split": split_name,
        "n": int(len(frame)),
        "mae": float(metrics.mae(frame["true_score"], frame["pred"])),
        "corr": float(metrics.pearson(frame["true_score"], frame["pred"])),
        "main_years": MAIN_YEARS,
        "main_n": int(len(main)),
        "main_mae": float(metrics.mae(main["true_score"], main["pred"])),
        "main_corr": float(metrics.pearson(main["true_score"], main["pred"])),
    }
    try:
        out["main_choice"] = float(metrics.choice_accuracy(main)["accuracy"])
    except Exception as exc:  # choice needs at least two candidates per group
        out["main_choice_error"] = str(exc)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-split", default="train_r")
    ap.add_argument("--eval-split", default="test_r")
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--suffix", default="")
    args = ap.parse_args()

    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean = clean.set_index("orig_row_id", drop=False)
    train = _load_split(config.V3_SPLITS / f"{args.train_split}.jsonl")
    rows = clean.loc[train["orig_row_id"]].reset_index(drop=True)

    X = _design(rows, CAT_COLS)
    y = train["true_score"].values

    model = HistGradientBoostingRegressor(
        max_iter=args.max_iter, learning_rate=0.08, max_bins=64,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=42,
    )
    model.fit(X, y)

    # sklearn stores the negative of its training loss. For squared_error the
    # per-sample loss is 0.5 * (y - pred)^2, so -2 * score is the MSE. The first
    # entry is the initial (mean) prediction; the rest are boosting stages.
    train_curve = [-2.0 * float(v) for v in model.train_score_]
    val_curve = [-2.0 * float(v) for v in model.validation_score_]
    best_idx = int(np.argmin(val_curve))

    out = {
        "train_split": args.train_split,
        "n_rows": int(len(y)),
        "n_features": int(X.shape[1]),
        "max_iter": args.max_iter,
        "n_iter_used": int(model.n_iter_),
        "early_stopped": bool(model.n_iter_ < args.max_iter),
        "loss": "MSE (converted from sklearn's negative half-MSE score)",
        "train_mse": train_curve,
        "validation_mse": val_curve,
        "final_train_mse": train_curve[-1],
        "final_validation_mse": val_curve[-1],
        "best_validation_mse": val_curve[best_idx],
        "best_tree": best_idx,
        "train_validation_gap_mse": val_curve[-1] - train_curve[-1],
    }
    out["test"] = _evaluate(model, clean, args.eval_split, X.columns)

    out_dir = Path(config.RESULTS) / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    sfx = args.suffix
    (out_dir / f"histgb_curve{sfx}.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )

    test = out["test"]
    lines = [
        "# HistGB training curve",
        "",
        f"- split: `{args.train_split}`, rows {out['n_rows']:,}, features {out['n_features']}",
        f"- budget {args.max_iter}; "
        + (f"early stopping triggered at {out['n_iter_used']} trees"
           if out["early_stopped"] else "early stopping did not trigger, the budget binds"),
        f"- final train MSE {out['final_train_mse']:.2f}, "
        f"final validation MSE {out['final_validation_mse']:.2f}",
        f"- best validation MSE {out['best_validation_mse']:.2f} at tree "
        f"{out['best_tree']}; train--validation gap "
        f"{out['train_validation_gap_mse']:.2f} MSE",
    ]
    if test:
        lines += [
            f"- `{test['split']}` all rows: MAE {test['mae']:.2f}, corr {test['corr']:.3f} "
            f"(n={test['n']:,})",
            f"- `{test['split']}` main years {MAIN_YEARS}: MAE {test['main_mae']:.2f}, "
            f"corr {test['main_corr']:.3f}, choice "
            f"{test.get('main_choice', float('nan')):.3f} (n={test['main_n']:,})",
        ]
    lines += ["", "| stage (0 = initial mean) | train MSE | validation MSE |",
              "|---:|---:|---:|",
              f"| 0 | {train_curve[0]:.2f} | {val_curve[0]:.2f} |"]
    for i, (tr, va) in enumerate(zip(train_curve[1:], val_curve[1:]), 1):
        lines.append(f"| {i} | {tr:.2f} | {va:.2f} |")
    (out_dir / f"histgb_curve{sfx}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.plot(range(len(train_curve)), train_curve, label="train")
        ax.plot(range(len(val_curve)), val_curve, label="validation (10%)")
        ax.axvline(best_idx, ls=":", lw=1, color="grey")
        ax.set_xlabel("boosting stage (0 = initial mean)")
        ax.set_ylabel("MSE")
        ax.set_title(f"HistGB budget {args.max_iter}, used {out['n_iter_used']} trees")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(out_dir / f"histgb_curve{sfx}.png", dpi=200)
    except Exception as exc:  # matplotlib optional
        print(f"[warn] plot skipped: {exc}")

    print(json.dumps({
        "n_iter_used": out["n_iter_used"],
        "early_stopped": out["early_stopped"],
        "final_validation_mse": round(out["final_validation_mse"], 2),
        "test": {k: (round(v, 3) if isinstance(v, float) else v)
                 for k, v in test.items()},
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
