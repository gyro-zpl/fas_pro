#!/usr/bin/env python
"""Review diagnostic: prior-only FAS baseline.

This script reconstructs the 2024 joint distribution Q* with the same code path
used to generate the GSS fine-tuning data, then samples synthetic respondents
*directly from Q* (no language model) and scores them against the empirical
2024 GSS sample with the same JS/KL functions used in the paper tables.

It is a review artifact: it does not modify the thesis or the experiment
results. Run it from the repository root, e.g.:

    python docs/03_进度与执行/prior_baseline_check.py \
        --data-dir /home/jpzhao_team/cyyuan/FAS_PRO/data/gss \
        --samples 400 --seed 42 --bootstrap 1000

Note: the pushed copy of this repository contains only the 2024 GSS samples;
the historical GSS 2004--2022 CSVs live in the original checkout
(FAS_PRO/data/gss).

The result should be compared with experiments/results/paper_tables/gss_fas.md.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "sft" / "teacher_distribution"))
sys.path.insert(0, str(ROOT / "codes" / "GSS"))

import get_distribution as gd  # noqa: E402
import evaluation as ev  # noqa: E402

CONFIGS = {
    "EDC": {
        "attrs": ["nateduc", "natdrug", "natcrime"],
        "real": "GSS2024_sampled_edc.csv",
    },
    "EHC": {
        "attrs": ["natenvir", "natheal", "natcity"],
        "real": "GSS2024_sampled_ehc.csv",
    },
}
HISTORICAL_YEARS = {2004, 2008, 2012, 2014, 2016, 2018, 2021, 2022}
LEVELS = ["1", "2", "3"]
TARGET_YEAR = 2024


def sample_from_joint(q: np.ndarray, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    flat = (q / q.sum()).flatten()
    idx = rng.choice(len(flat), size=n, replace=True, p=flat)
    rows = []
    for m in idx:
        i = m // 9
        j = (m % 9) // 3
        k = m % 3
        rows.append([LEVELS[i], LEVELS[j], LEVELS[k]])
    return pd.DataFrame(rows, columns=["a", "b", "c"])


def score(real: pd.DataFrame, gen: pd.DataFrame, attrs: list[str]) -> tuple[float, float]:
    gen = gen.copy()
    gen.columns = attrs
    gen = ev.clean_df(gen, attrs)
    marg = [ev.column_marginal_metrics(real, gen, c) for c in attrs]
    joints = [
        ev.pair_joint_metrics(real, gen, attrs[0], attrs[1]),
        ev.pair_joint_metrics(real, gen, attrs[0], attrs[2]),
        ev.pair_joint_metrics(real, gen, attrs[1], attrs[2]),
    ]
    return (
        float(np.mean([m["JS"] for m in marg])),
        float(np.mean([j["JS"] for j in joints])),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(ROOT / "data" / "gss"))
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=0,
                    help="optional bootstrap resamples of the generated rows")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    print(f"data dir: {data_dir}")
    print(f"samples: {args.samples}, seed: {args.seed}")
    print()
    print("| config | marginal JS | joint JS |")
    print("|---|---|---|")

    for name, cfg in CONFIGS.items():
        attrs = cfg["attrs"]
        q, *_ = gd.process_data_with_attrs(
            attr_names=attrs,
            data_dir=str(data_dir),
            target_years=HISTORICAL_YEARS,
            target_year=TARGET_YEAR,
        )
        real = ev.clean_df(pd.read_csv(data_dir / cfg["real"]), attrs)
        gen = sample_from_joint(q, args.samples, args.seed)
        marg, joint = score(real, gen, attrs)
        print(f"| {name} prior-only | {marg:.4f} | {joint:.4f} |")

        if args.bootstrap:
            rng = np.random.default_rng(args.seed)
            ms, js = [], []
            for _ in range(args.bootstrap):
                boot = gen.iloc[rng.integers(0, len(gen), len(gen))]
                m, j = score(real, boot, attrs)
                ms.append(m)
                js.append(j)
            mlo, mhi = np.percentile(ms, [2.5, 97.5])
            jlo, jhi = np.percentile(js, [2.5, 97.5])
            print(f"| {name} prior-only 95% CI | [{mlo:.4f}, {mhi:.4f}] | "
                  f"[{jlo:.4f}, {jhi:.4f}] |")

    print()
    print("Compare with experiments/results/paper_tables/gss_fas.md:")
    print("  EDC base 0.0552/0.1181; SFT 0.0048/0.0160; SFT-v3 0.0040/0.0215")
    print("  EHC base 0.0319/0.1001; SFT 0.0167/0.0470; SFT-v3 0.0077/0.0363")


if __name__ == "__main__":
    main()
