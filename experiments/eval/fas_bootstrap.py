"""Bootstrap 95% CIs for the FAS table (over generated rows).

Outputs:
    experiments/results/paper_tables/fas_bootstrap.json
    experiments/results/paper_tables/fas_bootstrap.md
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

from experiments import config

sys.path.insert(0, str(config.ROOT / "codes" / "GSS"))
import evaluation as ev  # noqa: E402

CONFIGS = {
    "EDC": {
        "cols": ["nateduc", "natdrug", "natcrime"],
        "real": config.GSS_DIR / "GSS2024_sampled_edc.csv",
        "runs": {
            "base": "gss_31_edc.csv",
            "SFT": "gss_31_sft_edc.csv",
            "SFT-v3": "gss_31_none_RAG_sft_edc_v3.csv",
        },
    },
    "EHC": {
        "cols": ["natenvir", "natheal", "natcity"],
        "real": config.GSS_DIR / "GSS2024_sampled_ehc.csv",
        "runs": {
            "base": "gss_31.csv",
            "SFT": "gss_31_sft.csv",
            "SFT-v3": "gss_31_none_RAG_sft_ehc_v3.csv",
        },
    },
}


def _stats(real, gen, cols):
    marg = [ev.column_marginal_metrics(real, gen, c)["JS"] for c in cols]
    joints = [
        ev.pair_joint_metrics(real, gen, cols[0], cols[1])["JS"],
        ev.pair_joint_metrics(real, gen, cols[0], cols[2])["JS"],
        ev.pair_joint_metrics(real, gen, cols[1], cols[2])["JS"],
    ]
    return float(np.mean(marg)), float(np.mean(joints))


def main() -> None:
    rng = np.random.default_rng(42)
    out = {}
    for name, cfg in CONFIGS.items():
        cols = cfg["cols"]
        real = ev.clean_df(pd.read_csv(cfg["real"]), cols)
        for label, fname in cfg["runs"].items():
            gen = ev.clean_df(pd.read_csv(config.GSS_RESULTS / fname), cols)
            point_m, point_j = _stats(real, gen, cols)
            ms, js = [], []
            n = len(gen)
            for _ in range(1000):
                idx = rng.integers(0, n, n)
                m, j = _stats(real, gen.iloc[idx], cols)
                ms.append(m)
                js.append(j)
            out[f"{name}|{label}"] = {
                "n": int(n),
                "marginal_js": point_m,
                "marginal_lo": float(np.percentile(ms, 2.5)),
                "marginal_hi": float(np.percentile(ms, 97.5)),
                "joint_js": point_j,
                "joint_lo": float(np.percentile(js, 2.5)),
                "joint_hi": float(np.percentile(js, 97.5)),
            }

    out_dir = config.RESULTS / "paper_tables"
    with open(out_dir / "fas_bootstrap.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = ["# FAS bootstrap 95% CIs (generated-row resampling)", "",
             "| config | run | n | marginal JS [95% CI] | joint JS [95% CI] |",
             "|---|---|---|---|---|"]
    for key, r in out.items():
        name, label = key.split("|")
        lines.append(
            f"| {name} | {label} | {r['n']} | "
            f"{r['marginal_js']:.4f} [{r['marginal_lo']:.4f}, {r['marginal_hi']:.4f}] | "
            f"{r['joint_js']:.4f} [{r['joint_lo']:.4f}, {r['joint_hi']:.4f}] |"
        )
    (out_dir / "fas_bootstrap.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
