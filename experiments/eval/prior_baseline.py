"""Prior-only FAS baseline: draw directly from the reconstructed joint Q*.

This makes the prior-only numbers in the thesis reproducible and persisted.
The prior is the same IPF reconstruction used to build the GSS fine-tuning
data; 400 synthetic respondents are sampled from Q* without a language model
and compared with the empirical 2024 GSS sample using the same JS/KL functions
as ``experiments/eval/gss_tables.py``.

Outputs:
    experiments/results/paper_tables/prior_baseline.json
    experiments/results/paper_tables/prior_baseline.md

The historical GSS CSVs (2004--2022) are not stored in this repository. Pass
``--data-dir`` to the directory that contains them, e.g.::

    python -m experiments.eval.prior_baseline \
        --data-dir /path/to/FAS_PRO/data/gss

The 2024 sampled CSVs must be in the same directory.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import config

sys.path.insert(0, str(config.ROOT / "sft" / "teacher_distribution"))
sys.path.insert(0, str(config.ROOT / "codes" / "GSS"))

import get_distribution as gd  # noqa: E402
import evaluation as ev  # noqa: E402

CONFIGS = {
    "EDC": {
        "attrs": ["nateduc", "natdrug", "natcrime"],
        "real": "GSS2024_sampled_edc.csv",
        "v3": "gss_31_none_RAG_sft_edc_v3.csv",
        "sft": "gss_31_sft_edc.csv",
        "base": "gss_31_edc.csv",
    },
    "EHC": {
        "attrs": ["natenvir", "natheal", "natcity"],
        "real": "GSS2024_sampled_ehc.csv",
        "v3": "gss_31_none_RAG_sft_ehc_v3.csv",
        "sft": "gss_31_sft.csv",
        "base": "gss_31.csv",
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


def score(real: pd.DataFrame, gen: pd.DataFrame, attrs: list[str]) -> dict:
    gen = gen.copy()
    gen.columns = attrs
    gen = ev.clean_df(gen, attrs)
    marg = [ev.column_marginal_metrics(real, gen, c) for c in attrs]
    joints = [
        ev.pair_joint_metrics(real, gen, attrs[0], attrs[1]),
        ev.pair_joint_metrics(real, gen, attrs[0], attrs[2]),
        ev.pair_joint_metrics(real, gen, attrs[1], attrs[2]),
    ]
    return {
        "n": int(len(gen)),
        "marginal_js": float(np.mean([m["JS"] for m in marg])),
        "joint_js": float(np.mean([j["JS"] for j in joints])),
        "marginal_kl_sym": float(np.mean([m["KL_symmetric"] for m in marg])),
        "joint_kl_sym": float(np.mean([j["KL_symmetric"] for j in joints])),
    }


def bootstrap(real: pd.DataFrame, gen: pd.DataFrame, attrs: list[str],
              n_boot: int, seed: int) -> dict:
    point = score(real, gen, attrs)
    rng = np.random.default_rng(seed)
    m_vals, j_vals = [], []
    for _ in range(n_boot):
        boot = gen.iloc[rng.integers(0, len(gen), len(gen))]
        r = score(real, boot, attrs)
        m_vals.append(r["marginal_js"])
        j_vals.append(r["joint_js"])
    return {
        **point,
        "marginal_ci": [float(np.percentile(m_vals, 2.5)),
                        float(np.percentile(m_vals, 97.5))],
        "joint_ci": [float(np.percentile(j_vals, 2.5)),
                     float(np.percentile(j_vals, 97.5))],
    }


def seed_sensitivity(q: np.ndarray, real: pd.DataFrame, attrs: list[str],
                     n: int, seeds: list[int]) -> dict:
    rows = {}
    for s in seeds:
        rows[str(s)] = score(real, sample_from_joint(q, n, s), attrs)
    out = {}
    for key in ["marginal_js", "joint_js"]:
        vals = [r[key] for r in rows.values()]
        out[key] = {
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "mean": float(np.mean(vals)),
            "values": [float(v) for v in vals],
        }
    return {"seeds": seeds, "metrics": out}


def compare(real: pd.DataFrame, prior_gen: pd.DataFrame, other_gen: pd.DataFrame,
            attrs: list[str], n_boot: int, seed: int) -> dict:
    """Independent bootstrap CI for prior minus other (negative = prior better)."""
    prior_point = score(real, prior_gen, attrs)
    other_point = score(real, other_gen, attrs)
    rng = np.random.default_rng(seed)
    dm, dj = [], []
    for _ in range(n_boot):
        p = prior_gen.iloc[rng.integers(0, len(prior_gen), len(prior_gen))]
        o = other_gen.iloc[rng.integers(0, len(other_gen), len(other_gen))]
        rp = score(real, p, attrs)
        ro = score(real, o, attrs)
        dm.append(rp["marginal_js"] - ro["marginal_js"])
        dj.append(rp["joint_js"] - ro["joint_js"])
    return {
        "other": other_point,
        "delta_marginal": prior_point["marginal_js"] - other_point["marginal_js"],
        "delta_marginal_ci": [float(np.percentile(dm, 2.5)),
                              float(np.percentile(dm, 97.5))],
        "delta_joint": prior_point["joint_js"] - other_point["joint_js"],
        "delta_joint_ci": [float(np.percentile(dj, 2.5)),
                           float(np.percentile(dj, 97.5))],
    }


def _load_gen(path: Path, attrs: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path)
    return ev.clean_df(df, attrs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=str(config.GSS_DIR),
                    help="directory containing historical and 2024 GSS CSVs")
    ap.add_argument("--samples", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--sample-seeds", default="42,43,44,45,46,47,48,49,50,51")
    args = ap.parse_args()
    data_dir = Path(args.data_dir)
    if not (data_dir / "GSS2004.csv").exists():
        raise SystemExit(
            f"historical GSS CSVs not found in {data_dir}; pass --data-dir")
    sample_seeds = [int(s) for s in args.sample_seeds.split(",") if s]

    out = {
        "data_dir": str(data_dir),
        "historical_years": sorted(HISTORICAL_YEARS),
        "target_year": TARGET_YEAR,
        "samples": args.samples,
        "bootstrap": args.bootstrap,
        "seed": args.seed,
        "sample_seeds": sample_seeds,
        "configs": {},
    }

    for name, cfg in CONFIGS.items():
        attrs = cfg["attrs"]
        print(f"\n=== {name}: reconstructing Q* ===")
        q, *_ = gd.process_data_with_attrs(
            attr_names=attrs,
            data_dir=str(data_dir),
            target_years=HISTORICAL_YEARS,
            target_year=TARGET_YEAR,
        )
        real = ev.clean_df(pd.read_csv(data_dir / cfg["real"]), attrs)
        prior_gen = sample_from_joint(q, args.samples, args.seed)
        prior = bootstrap(real, prior_gen, attrs, args.bootstrap, args.seed)
        sensitivity = seed_sensitivity(q, real, attrs, args.samples, sample_seeds)

        comparisons = {}
        for label in ["v3", "sft", "base"]:
            path = config.GSS_RESULTS / cfg[label]
            if path.exists():
                other = _load_gen(path, attrs)
                comparisons[label] = compare(
                    real, prior_gen, other, attrs, args.bootstrap, args.seed + 1)
            else:
                comparisons[label] = None

        out["configs"][name] = {
            "attrs": attrs,
            "real_file": cfg["real"],
            "n_real": int(len(real)),
            "q_joint": np.asarray(q).round(12).tolist(),
            "prior": prior,
            "seed_sensitivity": sensitivity,
            "comparisons": comparisons,
        }

    out_dir = config.RESULTS / "paper_tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "prior_baseline.json", "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    lines = [
        "# Prior-only FAS baseline",
        "",
        f"Reconstructed Q* from {sorted(HISTORICAL_YEARS)}; target {TARGET_YEAR}; "
        f"{args.samples} draws; base seed {args.seed}; "
        f"{args.bootstrap:,} bootstrap resamples of the generated rows.",
        "",
        "| Config | Prior marginal JS [95% CI] | Prior joint JS [95% CI] | "
        "Seed range marginal | Seed range joint |",
        "|---|---|---|---|---|",
    ]
    for name, r in out["configs"].items():
        p = r["prior"]
        s = r["seed_sensitivity"]["metrics"]
        lines.append(
            f"| {name} | {p['marginal_js']:.4f} "
            f"[{p['marginal_ci'][0]:.4f}, {p['marginal_ci'][1]:.4f}] | "
            f"{p['joint_js']:.4f} [{p['joint_ci'][0]:.4f}, {p['joint_ci'][1]:.4f}] | "
            f"{s['marginal_js']['min']:.4f}--{s['marginal_js']['max']:.4f} | "
            f"{s['joint_js']['min']:.4f}--{s['joint_js']['max']:.4f} |"
        )

    lines += [
        "",
        "## Comparison with the LLM runs",
        "",
        "Delta is prior minus LLM; negative means the prior is closer to the "
        "empirical 2024 sample. CIs resample each run independently.",
        "",
        "| Config | vs | Delta marginal JS [95% CI] | Delta joint JS [95% CI] |",
        "|---|---|---|---|",
    ]
    for name, r in out["configs"].items():
        for label, c in r["comparisons"].items():
            if c is None:
                continue
            lines.append(
                f"| {name} | {label} | {c['delta_marginal']:+.4f} "
                f"[{c['delta_marginal_ci'][0]:+.4f}, {c['delta_marginal_ci'][1]:+.4f}] | "
                f"{c['delta_joint']:+.4f} "
                f"[{c['delta_joint_ci'][0]:+.4f}, {c['delta_joint_ci'][1]:+.4f}] |"
            )

    lines += [
        "",
        "## Reproduce",
        "",
        "```bash",
        "python -m experiments.eval.prior_baseline \\",
        f"    --data-dir {data_dir} --samples {args.samples} "
        f"--seed {args.seed} --bootstrap {args.bootstrap}",
        "```",
    ]
    (out_dir / "prior_baseline.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
