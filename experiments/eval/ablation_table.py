"""Build the ablation table (in-distribution 2016-2019 and 2024).

Outputs:
    experiments/results/paper_tables/ablation.json
    experiments/results/paper_tables/ablation.md
"""
from __future__ import annotations

import json
import statistics

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics

CONFIGS = {
    "Full (9-strategy)": [
        "qwen3_t_seed42_test_r_generate",
        "qwen3_t_seed43_test_r_generate",
        "qwen3_t_seed44_test_r_generate",
    ],
    "A1 random sampling": ["a1_random"],
    "A2 minimal prompt": ["a2_minimal"],
    "A3 name-only prompt": ["a3_name_only"],
}
IN_DIST = [2016, 2018, 2019]


def _metrics(df: pd.DataFrame) -> dict:
    corr = float(np.corrcoef(df["true_score"], df["pred_float"])[0, 1]) if df["pred_float"].std() > 1e-9 else float("nan")
    ch = metrics.choice_accuracy(df)
    return {
        "n": int(len(df)),
        "mae": metrics.mae(df["true_score"], df["pred_float"]),
        "corr": corr,
        "choice": ch["accuracy"],
        "chance": ch["chance"],
    }


def main() -> None:
    out = {}
    for label, stems in CONFIGS.items():
        rec = {"in_dist": [], "2024": []}
        for stem in stems:
            path = config.PREDS / f"{stem}.predictions.jsonl"
            if not path.exists():
                continue
            df = pd.read_json(path, lines=True)
            rec["in_dist"].append(_metrics(df[df["year"].isin(IN_DIST)]))
            rec["2024"].append(_metrics(df[df["year"] == 2024]))
        entry = {}
        for split in ["in_dist", "2024"]:
            vals = rec[split]
            if not vals:
                continue
            entry[split] = {}
            for key in ["n", "mae", "corr", "choice", "chance"]:
                xs = [v[key] for v in vals if v[key] == v[key]]
                entry[split][key] = statistics.mean(xs) if xs else float("nan")
                entry[split][f"{key}_std"] = statistics.pstdev(xs) if len(xs) > 1 else 0.0
        out[label] = entry

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "ablation.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Ablations", "",
             "| config | split | n | MAE | corr | choice | chance |",
             "|---|---|---|---|---|---|---|"]
    for label, entry in out.items():
        for split, r in entry.items():
            mae = f"{r['mae']:.2f}" + (f" ± {r['mae_std']:.2f}" if r["mae_std"] else "")
            corr = f"{r['corr']:+.3f}" + (f" ± {r['corr_std']:.3f}" if r["corr_std"] else "")
            choice = f"{r['choice']:.3f}" + (f" ± {r['choice_std']:.3f}" if r["choice_std"] else "")
            lines.append(f"| {label} | {split} | {r['n']} | {mae} | {corr} | {choice} | {r['chance']:.3f} |")
    (out_dir / "ablation.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
