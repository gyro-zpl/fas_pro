"""Generate the paper's GSS FAS table from saved generation CSVs.

Outputs:
    experiments/results/paper_tables/gss_fas.json
    experiments/results/paper_tables/gss_fas.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

from experiments import config

sys.path.insert(0, str(config.ROOT / "codes" / "GSS"))
import evaluation as ev  # noqa: E402

CONFIGS = {
    "EHC": {
        "cols": ["natenvir", "natheal", "natcity"],
        "real": config.GSS_DIR / "GSS2024_sampled_ehc.csv",
        "runs": {
            "base": "gss_31.csv",
            "sft": "gss_31_sft.csv",
            "RAG": "gss_31_RAG.csv",
            "RAG+sft": "gss_31_RAG_sft.csv",
            "noneRAG+sft-v2": "gss_31_none_RAG_sft_ehc_v2.csv",
            "noneRAG+sft-v3": "gss_31_none_RAG_sft_ehc_v3.csv",
            "quota": "gss_quota_ehc.csv",
        },
    },
    "EDC": {
        "cols": ["nateduc", "natdrug", "natcrime"],
        "real": config.GSS_DIR / "GSS2024_sampled_edc.csv",
        "runs": {
            "base": "gss_31_edc.csv",
            "sft": "gss_31_sft_edc.csv",
            "RAG": "gss_31_RAG_edc.csv",
            "RAG+sft": "gss_31_RAG_sft_edc.csv",
            "noneRAG+sft-v2": "gss_31_none_RAG_sft_edc_v2.csv",
            "noneRAG+sft-v3": "gss_31_none_RAG_sft_edc_v3.csv",
            "quota": "gss_quota_edc.csv",
        },
    },
}


def main() -> None:
    out_dir = config.RESULTS / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    table = {}
    for name, cfg in CONFIGS.items():
        cols = cfg["cols"]
        real = ev.clean_df(pd.read_csv(cfg["real"]), cols)
        for label, fname in cfg["runs"].items():
            path = config.GSS_RESULTS / fname
            if not path.exists():
                continue
            gen = ev.clean_df(pd.read_csv(path), cols)
            marg = [ev.column_marginal_metrics(real, gen, c) for c in cols]
            joints = [
                ev.pair_joint_metrics(real, gen, cols[0], cols[1]),
                ev.pair_joint_metrics(real, gen, cols[0], cols[2]),
                ev.pair_joint_metrics(real, gen, cols[1], cols[2]),
            ]
            table[f"{name}|{label}"] = {
                "n_gen": int(len(gen)),
                "marginal_js": sum(m["JS"] for m in marg) / len(marg),
                "marginal_kl_sym": sum(m["KL_symmetric"] for m in marg) / len(marg),
                "joint_js": sum(j["JS"] for j in joints) / len(joints),
                "joint_kl_sym": sum(j["KL_symmetric"] for j in joints) / len(joints),
            }

    with open(out_dir / "gss_fas.json", "w") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)

    lines = ["# GSS FAS", "",
             "| config | run | n | marginal JS | marginal KL | joint JS | joint KL |",
             "|---|---|---|---|---|---|---|"]
    for key, r in table.items():
        name, label = key.split("|")
        lines.append(f"| {name} | {label} | {r['n_gen']} | {r['marginal_js']:.4f} | "
                     f"{r['marginal_kl_sym']:.4f} | {r['joint_js']:.4f} | {r['joint_kl_sym']:.4f} |")
    (out_dir / "gss_fas.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
