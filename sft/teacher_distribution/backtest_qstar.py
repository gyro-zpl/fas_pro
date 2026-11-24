"""Back-test the FAS reconstruction: rebuild a historical year from earlier waves.

For a cutoff year Y, this script:
  1. takes every GSS wave strictly before Y,
  2. extrapolates the three marginals and the B-C table to Y,
  3. fuses them with the same IPF as the main pipeline, giving Q*,
  4. scores Q* against the *actual* Y wave, which the pipeline never saw.

Two references are reported so the number is interpretable:
  - `persistence`: reuse the last wave before Y as the prediction;
  - `independence`: three marginals multiplied, i.e. no B-C association.

Outputs:
    experiments/results/paper_tables/qstar_backtest_<Y>.json
    experiments/results/paper_tables/qstar_backtest.md   (appended)

Usage:
    python sft/teacher_distribution/backtest_qstar.py --cutoff 2018
    python sft/teacher_distribution/backtest_qstar.py --cutoff 2022
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import get_distribution as gd  # noqa: E402

LEVELS = ["1", "2", "3"]


def js(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=float).ravel()
    q = np.asarray(q, dtype=float).ravel()
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def empirical(df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    idx = {lv: i for i, lv in enumerate(LEVELS)}
    Q = np.zeros((3, 3, 3))
    for _, r in df.iterrows():
        Q[idx[str(r[gd.ATTR_A]).strip()], idx[str(r[gd.ATTR_B]).strip()], idx[str(r[gd.ATTR_C]).strip()]] += 1
    Q = Q / Q.sum()
    return Q, {
        "marginal": [Q.sum(axis=(1, 2)), Q.sum(axis=(0, 2)), Q.sum(axis=(0, 1))],
        "pair": [Q.sum(axis=2), Q.sum(axis=1), Q.sum(axis=0)],
    }


def scores(Q: np.ndarray, real: np.ndarray, real_parts: dict) -> dict:
    marg = [Q.sum(axis=(1, 2)), Q.sum(axis=(0, 2)), Q.sum(axis=(0, 1))]
    pair = [Q.sum(axis=2), Q.sum(axis=1), Q.sum(axis=0)]
    return {
        "marginal_js": float(np.mean([js(a, b) for a, b in zip(marg, real_parts["marginal"])])),
        "pair_js": float(np.mean([js(a, b) for a, b in zip(pair, real_parts["pair"])])),
        "three_way_js": float(js(Q, real)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cutoff", type=int, default=2018)
    ap.add_argument("--data-dir", default=gd.DATA_DIR)
    ap.add_argument("--attrs", nargs=3, default=None,
                    help="A B C attribute names, default = module setting")
    args = ap.parse_args()

    gd.DATA_DIR = args.data_dir
    if args.attrs:
        gd.set_attr_names(*args.attrs)

    df = gd.load_and_process_data()
    stats = gd.year_stats(df)

    before = {y: s for y, s in stats.items() if y < args.cutoff}
    if len(before) < 4:
        raise SystemExit(f"need >= 4 waves before {args.cutoff}, got {sorted(before)}")

    # reconstruction with the production recipe
    Pa, Pb, Pc, Pbc = gd.project_to_year_linear(before, args.cutoff)
    Q_star = gd.ipf_joint(Pa, Pb, Pc, Pbc, iters=300)

    # references
    last_year = max(before)
    s_last = before[last_year]
    Q_persist = np.outer(np.asarray(s_last[f"P_{gd.ATTR_A}"]), np.asarray(s_last[f"P_{gd.ATTR_B}_{gd.ATTR_C}"])).reshape(3, 3, 3)
    Q_indep = np.multiply.outer(np.asarray(Pa), np.asarray(Pb))
    Q_indep = np.multiply.outer(Q_indep.reshape(3, 3), np.asarray(Pc)).reshape(3, 3, 3)
    Q_indep /= Q_indep.sum()

    real_df = df[df.year == args.cutoff]
    if real_df.empty:
        raise SystemExit(f"no rows for {args.cutoff}")
    real, parts = empirical(real_df)

    report = {
        "cutoff": args.cutoff,
        "waves_used": sorted(before),
        "persistence_from": last_year,
        "n_real": int(len(real_df)),
        "qstar": scores(Q_star, real, parts),
        "persistence": scores(Q_persist, real, parts),
        "independence": scores(Q_indep, real, parts),
        "internal": {
            "A_marginal_gap": float(np.abs(np.asarray(Pa) - Q_star.sum(axis=(1, 2))).sum()),
            "B_marginal_gap": float(np.abs(np.asarray(Pb) - Q_star.sum(axis=(0, 2))).sum()),
            "C_marginal_gap": float(np.abs(np.asarray(Pc) - Q_star.sum(axis=(0, 1))).sum()),
            "BC_gap": float(np.abs(np.asarray(Pbc) - Q_star.sum(axis=0)).sum()),
        },
    }

    out_dir = Path("experiments/results/paper_tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"qstar_backtest_{args.cutoff}.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    lines = [
        f"## Q* back-test at {args.cutoff}",
        "",
        f"- waves used: {report['waves_used']}",
        f"- real rows at {args.cutoff}: {report['n_real']}",
        "",
        "| reconstruction | marginal JS | pairwise JS | three-way JS |",
        "|---|---:|---:|---:|",
    ]
    for name in ("qstar", "persistence", "independence"):
        s = report[name]
        lines.append(f"| {name} | {s['marginal_js']:.4f} | {s['pair_js']:.4f} | {s['three_way_js']:.4f} |")
    lines += [
        "",
        "Internal consistency of Q* (L1 gaps, should be ~0 for the constrained terms):",
        "",
        f"- A marginal gap {report['internal']['A_marginal_gap']:.4f}",
        f"- B marginal gap {report['internal']['B_marginal_gap']:.4f}",
        f"- C marginal gap {report['internal']['C_marginal_gap']:.4f}",
        f"- B-C joint gap {report['internal']['BC_gap']:.4f}",
        "",
    ]
    md = out_dir / "qstar_backtest.md"
    with md.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
