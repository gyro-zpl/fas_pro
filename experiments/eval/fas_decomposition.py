"""Decompose the FAS error into "can't sample Q*" and "Q* is off".

For every generated sample we report three distances:
  - to Q* (the reconstructed teacher distribution, 27 cells from
    `prior_baseline.json`): how far the model is from what it was taught;
  - to the real 2024 sample (numbers copied from `gss_fas.md`): the headline
    metric used in the thesis;
  - the gap between the two is the part of the error that Q* itself owns.

Reading rule: if distance-to-Q* dominates, the model is the bottleneck; if it
is near zero while the real-2024 distance stays high, Q* is the bottleneck.

Outputs:
    experiments/results/paper_tables/fas_decomposition.md

Usage:
    python -m experiments.eval.fas_decomposition
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# (config, run label, csv in data/gss/test_results)
RUNS = [
    ("EDC", "base", "gss_31_edc.csv"),
    ("EDC", "SFT", "gss_31_sft_edc.csv"),
    ("EDC", "SFT-v3", "gss_31_none_RAG_sft_edc_v3.csv"),
    ("EDC", "RAG", "gss_31_RAG_edc.csv"),
    ("EHC", "base", "gss_31_ehc.csv"),
    ("EHC", "SFT", "gss_31_sft_ehc.csv"),
    ("EHC", "SFT-v3", "gss_31_none_RAG_sft_ehc_v3.csv"),
]

# headline distances to the real 2024 sample, from gss_fas.md
VS_REAL = {
    ("EDC", "base"): (0.0552, 0.1181),
    ("EDC", "SFT"): (0.0048, 0.0160),
    ("EDC", "SFT-v3"): (0.0040, 0.0215),
    ("EDC", "RAG"): (0.0473, 0.1096),
    ("EHC", "base"): (0.0319, 0.1001),
    ("EHC", "SFT"): (0.0167, 0.0470),
    ("EHC", "SFT-v3"): (0.0077, 0.0363),
    ("EHC", "RAG"): (0.0404, 0.1095),
}


def js(p, q) -> float:
    p = np.asarray(p, float).ravel()
    q = np.asarray(q, float).ravel()
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def dist_to_q(df: pd.DataFrame, Q: np.ndarray) -> tuple[float, float, float]:
    emp = np.zeros((3, 3, 3))
    for _, r in df.iterrows():
        emp[int(r.iloc[0]) - 1, int(r.iloc[1]) - 1, int(r.iloc[2]) - 1] += 1
    emp /= emp.sum()
    marg = np.mean([
        js(emp.sum(axis=(1, 2)), Q.sum(axis=(1, 2))),
        js(emp.sum(axis=(0, 2)), Q.sum(axis=(0, 2))),
        js(emp.sum(axis=(0, 1)), Q.sum(axis=(0, 1))),
    ])
    pair = np.mean([
        js(emp.sum(axis=2), Q.sum(axis=2)),
        js(emp.sum(axis=1), Q.sum(axis=1)),
        js(emp.sum(axis=0), Q.sum(axis=0)),
    ])
    return float(marg), float(pair), float(js(emp, Q))


def main() -> None:
    prior = json.loads(Path("experiments/results/paper_tables/prior_baseline.json").read_text())
    rows = []
    for cfg, run, fname in RUNS:
        path = Path("data/gss/test_results") / fname
        if not path.exists():
            print(f"[skip] {cfg} {run}: {fname} not in repo")
            continue
        Q = np.asarray(prior["configs"][cfg]["q_joint"], float)
        Q = Q / Q.sum()
        df = pd.read_csv(path, on_bad_lines="skip")
        if df.shape[1] != 3:
            print(f"[skip] {cfg} {run}: {fname} is not a 3-column sample file")
            continue
        m_q, p_q, t_q = dist_to_q(df, Q)
        m_r, p_r = VS_REAL.get((cfg, run), (float("nan"), float("nan")))
        rows.append((cfg, run, len(df), m_q, p_q, t_q, m_r, p_r))

    lines = [
        "# FAS error decomposition",
        "",
        "Distance to Q* uses the same JS definitions as the thesis (mean over the",
        "three marginals and the three pairwise joints); the three-way column is the",
        "full 27-cell divergence. The real-2024 columns are copied from `gss_fas.md`.",
        "",
        "| config | run | n | to Q*: marginal | to Q*: pairwise | to Q*: 3-way | to real: marginal | to real: pairwise |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for cfg, run, n, m_q, p_q, t_q, m_r, p_r in rows:
        lines.append(
            f"| {cfg} | {run} | {n} | {m_q:.4f} | {p_q:.4f} | {t_q:.4f} | {m_r:.4f} | {p_r:.4f} |"
        )
    lines += [
        "",
        "The prior's own distance to the real 2024 sample is 0.0004 / 0.0076 (EDC)",
        "and 0.0011 / 0.0225 (EHC). If a run's distance to Q* is larger than that,",
        "most of its error is generation, not reconstruction.",
        "",
    ]
    out = Path("experiments/results/paper_tables/fas_decomposition.md")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
