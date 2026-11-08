"""Cross-pressure subgroup analysis (in-distribution 2016-2019).

Outputs:
    experiments/results/paper_tables/cross_pressure.json
    experiments/results/paper_tables/cross_pressure.md
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from experiments import config
from experiments.eval import metrics

IN_DIST = [2016, 2018, 2019]
KEY = ["respondent_id", "year", "candidate"]


def _cp_table() -> pd.DataFrame:
    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean["pm"] = pd.to_numeric(clean["party_match"], errors="coerce")
    clean["idist"] = pd.to_numeric(clean["ideology_distance_7"], errors="coerce")
    clean["sim"] = pd.to_numeric(clean["demographic_similarity"], errors="coerce")
    clean["cp1"] = (clean["pm"] == 1) & (clean["idist"] >= 3)
    clean["cp2"] = (clean["pm"] == 0) & (clean["idist"] <= 1)
    clean["cp3"] = (clean["sim"] >= 0.7) & (clean["pm"] == 0)
    clean["cp_type"] = np.select(
        [clean["cp1"], clean["cp2"], clean["cp3"]], ["CP1", "CP2", "CP3"], default="none"
    )
    return clean[KEY + ["cp_type"]].drop_duplicates(KEY)


def main() -> None:
    cp = _cp_table()
    configs = {
        "llm_seed42": "qwen3_t_seed42_test_r_generate",
        "llm_seed43": "qwen3_t_seed43_test_r_generate",
        "llm_seed44": "qwen3_t_seed44_test_r_generate",
        "histgb": "tabular_histgb_test_r",
    }
    out = {}
    for name, stem in configs.items():
        df = pd.read_json(config.PREDS / f"{stem}.predictions.jsonl", lines=True)
        df = df[df["year"].isin(IN_DIST)].merge(cp, on=KEY, how="left")
        entry = {}
        for cp_type, g in df.groupby("cp_type"):
            corr = float(np.corrcoef(g["true_score"], g["pred_float"])[0, 1]) if g["pred_float"].std() > 1e-9 else float("nan")
            entry[cp_type] = {
                "n": int(len(g)),
                "mae": metrics.mae(g["true_score"], g["pred_float"]),
                "corr": corr,
            }
        out[name] = entry

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "cross_pressure.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# Cross-pressure subgroups (2016-2019)", "",
             "| config | CP1 | CP2 | CP3 | none |", "|---|---|---|---|---|"]
    for name, entry in out.items():
        cells = []
        for t in ["CP1", "CP2", "CP3", "none"]:
            r = entry.get(t)
            cells.append(f"n={r['n']}, MAE={r['mae']:.2f}, r={r['corr']:+.3f}" if r else "—")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    (out_dir / "cross_pressure.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
