"""Tabular baselines on the same features as the LLM.

If a gradient-boosted model on the raw features matches the LLM, the LLM's
contribution is not raw predictive power but generalization / distributional
behavior. This is a required sanity check.

Usage:
    python -m experiments.baselines.tabular
"""
from __future__ import annotations

import os

# Keep OpenMP from spawning one thread per core (the host has ~176; sklearn
# deadlocks/thrashes above a modest pool).
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments import config
from experiments.eval import metrics

ID_COLS = {"respondent_id", "orig_row_id", "support_score", "is_missing", "candidate_raw"}
CAT_COLS = ["candidate", "year", "party", "position_encoded", "gender_resp",
            "race_resp", "educ", "ideo5", "pid7", "party_match"]


def _load_split(path: Path) -> pd.DataFrame:
    rows = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            meta = obj["metadata"]
            rows.append({
                "orig_row_id": meta["orig_row_id"],
                "respondent_id": meta["respondent_id"],
                "year": meta["year"],
                "candidate": meta["candidate"],
                "true_score": meta["true_score"],
            })
    return pd.DataFrame(rows)


def _design(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    num_cols = [c for c in df.columns
                if c not in ID_COLS and c not in cat_cols
                and pd.api.types.is_numeric_dtype(df[c])]
    X = df[num_cols].copy()
    cats = [c for c in cat_cols if c in df.columns]
    if cats:
        dummies = pd.get_dummies(df[cats].astype(str), dummy_na=False)
        X = pd.concat([X, dummies], axis=1)
    return X


def main() -> None:
    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean = clean.set_index("orig_row_id", drop=False)

    # Fit once on train_t, then predict every split.
    train = _load_split(config.V3_SPLITS / "train_t.jsonl")
    train_rows = clean.loc[train["orig_row_id"]].reset_index(drop=True)
    X_train = _design(train_rows, CAT_COLS)
    y_train = train["true_score"].values

    models = {
        "histgb": HistGradientBoostingRegressor(
            max_iter=60, learning_rate=0.08, max_bins=64,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=5, random_state=42,
        ),
        "ridge": make_pipeline(
            SimpleImputer(strategy="median"), StandardScaler(),
            Ridge(alpha=1.0, solver="lsqr"),
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)

    results: dict[str, dict] = {}
    for split_name in ["val_t", "test_t", "test_r", "test_tr"]:
        path = config.V3_SPLITS / f"{split_name}.jsonl"
        if not path.exists():
            continue
        split = _load_split(path)
        rows = clean.loc[split["orig_row_id"]].reset_index(drop=True)
        X = _design(rows, CAT_COLS).reindex(columns=X_train.columns, fill_value=0)

        for name, model in models.items():
            pred = model.predict(X)
            frame = split.copy()
            frame["pred"] = np.clip(pred, 0, 100)
            frame["pred_float"] = frame["pred"]
            per_year = {}
            for year, g in frame.groupby("year"):
                per_year[str(int(year))] = {
                    "n": int(len(g)),
                    "mae": metrics.mae(g["true_score"], g["pred"]),
                    "corr": metrics.pearson(g["true_score"], g["pred"]),
                    "choice": metrics.choice_accuracy(g),
                }
            results[f"{name}_{split_name}"] = {
                "overall": metrics.summarize(frame),
                "per_year": per_year,
                "n": int(len(frame)),
            }
            frame.to_json(
                config.PREDS / f"tabular_{name}_{split_name}.predictions.jsonl",
                orient="records", lines=True,
            )

    out = config.RESULTS / "01_tabular.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    lines = ["# Tabular baselines", "",
             "| model | split | MAE | RMSE | relaxed ACC | choice ACC | chance |",
             "|---|---|---|---|---|---|---|"]
    for key, val in results.items():
        o = val["overall"]
        lines.append(
            f"| {key} | {val['n']} | {o['mae']:.3f} | {o['rmse']:.3f} | "
            f"{o['relaxed_acc']:.4f} | {o['choice']['accuracy']:.4f} | "
            f"{o['choice']['chance']:.4f} |"
        )
    (config.RESULTS / "01_tabular.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
