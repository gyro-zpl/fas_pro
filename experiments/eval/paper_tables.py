"""Generate the paper's PAS tables from saved predictions.

Outputs:
    experiments/results/paper_tables/pas_by_year.json
    experiments/results/paper_tables/pas_by_year.md

Rows are (model, split, year). LLM metrics are mean ± std over seeds.
"""
from __future__ import annotations

import glob
import json
import os
import re
import statistics
from collections import defaultdict

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics


def _year_metrics(df: pd.DataFrame) -> dict:
    out = {}
    for year, g in df.groupby("year"):
        corr = float(np.corrcoef(g["true_score"], g["pred_float"])[0, 1]) if g["pred_float"].std() > 1e-9 else float("nan")
        ch = metrics.choice_accuracy(g)
        out[int(year)] = {
            "n": int(len(g)),
            "mae": metrics.mae(g["true_score"], g["pred_float"]),
            "rmse": metrics.rmse(g["true_score"], g["pred_float"]),
            "corr": corr,
            "choice": ch["accuracy"],
            "chance": ch["chance"],
            "pred_std": float(g["pred_float"].std()),
        }
    return out


def _collect_llm() -> dict:
    rows = []
    for path in sorted(glob.glob(str(config.PREDS / "qwen3_t_seed*_test_r_generate.predictions.jsonl"))):
        seed = re.search(r"seed(\d+)", path).group(1)
        df = pd.read_json(path, lines=True)
        for year, m in _year_metrics(df).items():
            rows.append({"model": "llm_ft", "seed": seed, "split": "test_r", "year": year, **m})
    for path in sorted(glob.glob(str(config.PREDS / "qwen3_t_seed*_test_t_generate.predictions.jsonl"))):
        seed = re.search(r"seed(\d+)", path).group(1)
        df = pd.read_json(path, lines=True)
        for year, m in _year_metrics(df).items():
            rows.append({"model": "llm_ft", "seed": seed, "split": "test_t", "year": year, **m})
    base = config.PREDS / "qwen3_base_test_r_generate.predictions.jsonl"
    if base.exists():
        df = pd.read_json(base, lines=True)
        for year, m in _year_metrics(df).items():
            rows.append({"model": "llm_base", "seed": "-", "split": "test_r", "year": year, **m})
    return rows


def _collect_tabular() -> dict:
    rows = []
    for name in ["histgb", "ridge"]:
        for split in ["test_r", "test_t"]:
            path = config.PREDS / f"tabular_{name}_{split}.predictions.jsonl"
            if not path.exists():
                continue
            df = pd.read_json(path, lines=True)
            for year, m in _year_metrics(df).items():
                rows.append({"model": name, "seed": "-", "split": split, "year": year, **m})
    return rows


def _aggregate(rows: list[dict]) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["model"], r["split"], r["year"])].append(r)
    out = {}
    for (model, split, year), vals in sorted(groups.items()):
        rec = {"model": model, "split": split, "year": year, "n_seeds": len(vals)}
        for key in ["n", "mae", "rmse", "corr", "choice", "chance", "pred_std"]:
            xs = [v[key] for v in vals if v[key] == v[key]]  # drop NaN
            rec[key] = statistics.mean(xs) if xs else float("nan")
            rec[f"{key}_std"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        out[f"{model}|{split}|{year}"] = rec
    return out


def main() -> None:
    out_dir = config.RESULTS / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _collect_llm() + _collect_tabular()
    table = _aggregate(rows)
    with open(out_dir / "pas_by_year.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    lines = ["# PAS by year", "",
             "| model | split | year | n | n_seeds | MAE | corr | choice | chance | pred std |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in sorted(table.items(), key=lambda kv: (kv[1]["model"], kv[1]["split"], kv[1]["year"])):
        choice = f"{r['choice']:.3f}" + (f" ± {r['choice_std']:.3f}" if r["n_seeds"] > 1 else "")
        mae = f"{r['mae']:.2f}" + (f" ± {r['mae_std']:.2f}" if r["n_seeds"] > 1 else "")
        corr = f"{r['corr']:+.3f}" + (f" ± {r['corr_std']:.3f}" if r["n_seeds"] > 1 else "")
        lines.append(f"| {r['model']} | {r['split']} | {r['year']} | {r['n']} | {r['n_seeds']} "
                     f"| {mae} | {corr} | {choice} | {r['chance']:.3f} | {r['pred_std']:.1f} |")
    (out_dir / "pas_by_year.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
