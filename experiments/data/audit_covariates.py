"""Audit covariate availability and coding consistency across ANES waves.

Read-only. Answers three questions that the per-year analysis depends on:

1. Which model-visible covariates are present in each wave?
2. How many distinct political profiles does a wave actually contain after the
   prompt builder's default substitution?
3. Is `pid7` coded in the same direction in every wave?

Outputs:
    experiments/results/00_covariate_audit.md
    experiments/results/00_covariate_audit.json

Usage:
    python -m experiments.data.audit_covariates
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from experiments import config
from experiments.data import prompts

WAVES = [2016, 2018, 2019, 2020, 2022, 2024]
COVARIATES = ["pid7", "ideo5", "lcself", "educ", "gender_resp", "race_resp", "birthyr_resp"]
SPLITS = ["train_r", "test_r", "train_t", "test_t"]


def _blank(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    return series.isna() | (s == "") | (s == "nan") | (s == "None")


def _political_line(row) -> str:
    """The two prompt lines that carry partisanship and ideology."""
    text = prompts.build_candidate_prompt(row, level=config.PROMPT_LEVEL)
    keep = [
        ln for ln in text.splitlines()
        if ln.startswith("- Political:") or ln.startswith("- Party alignment:")
    ]
    return " | ".join(keep)


def main() -> None:
    df = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    df["pid7"] = pd.to_numeric(df["pid7"], errors="coerce")
    df["lcself"] = pd.to_numeric(df["lcself"], errors="coerce")
    df["party_match"] = pd.to_numeric(df["party_match"], errors="coerce")
    df["support_score"] = pd.to_numeric(df["support_score"], errors="coerce")

    report: dict = {"missing_share": {}, "sentinels": {}, "prompt_profiles": {},
                    "pid7_direction": {}, "party_match_corr": {}, "splits": {}}

    for wave in WAVES:
        sub = df[df.year == wave]
        report["missing_share"][wave] = {
            c: round(float(_blank(sub[c]).mean()), 4) for c in COVARIATES
        }
        # Sentinel / out-of-range codes that are not valid scale values.
        pid = sub["pid7"].dropna()
        report["sentinels"][wave] = {
            "pid7_value_counts": {str(int(k)): int(v) for k, v in pid.value_counts().items()},
            "n_out_of_range": int(((pid < 1) | (pid > 7)).sum()),
        }
        # Distinct political profiles a model can actually see.
        profiles = sub.apply(_political_line, axis=1).unique()
        report["prompt_profiles"][wave] = {
            "n_rows": int(len(sub)),
            "n_distinct_profiles": int(len(profiles)),
            "examples": list(profiles[:3]),
        }
        # Direction check: pid7 should correlate positively with liberal-conservative
        # self-placement in every wave.
        ok = sub[sub["pid7"].between(1, 7) & sub["lcself"].between(1, 7)]
        report["pid7_direction"][wave] = (
            round(float(ok["pid7"].corr(ok["lcself"])), 4) if len(ok) > 10 else None
        )
        # Sign check: co-partisans should rate higher.
        ok2 = sub.dropna(subset=["party_match", "support_score"])
        report["party_match_corr"][wave] = (
            round(float(ok2["party_match"].corr(ok2["support_score"])), 4) if len(ok2) > 10 else None
        )

    for split in SPLITS:
        path = config.V3_SPLITS / f"{split}.jsonl"
        if not path.exists():
            continue
        years: dict[int, int] = {}
        for line in path.open():
            y = json.loads(line)["metadata"]["year"]
            years[y] = years.get(y, 0) + 1
        report["splits"][split] = {str(k): v for k, v in sorted(years.items())}

    out_dir = Path(config.RESULTS)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "00_covariate_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Covariate availability audit",
        "",
        "Fraction of rows whose value is blank, and the number of distinct political",
        "profiles a model can actually see after the prompt builder substitutes defaults.",
        "",
        "## Missing share by wave",
        "",
        "| Wave | " + " | ".join(COVARIATES) + " |",
        "|---|" + "---:|" * len(COVARIATES),
    ]
    for wave in WAVES:
        cells = [f"{report['missing_share'][wave][c] * 100:.0f}%" for c in COVARIATES]
        lines.append(f"| {wave} | " + " | ".join(cells) + " |")

    lines += ["", "## Prompt-visible political profiles", "",
              "| Wave | rows | distinct profiles |", "|---|---:|---:|"]
    for wave in WAVES:
        p = report["prompt_profiles"][wave]
        lines.append(f"| {wave} | {p['n_rows']:,} | {p['n_distinct_profiles']} |")

    lines += ["", "## pid7 direction and party-match sign", "",
              "| Wave | corr(pid7, lcself) | corr(party_match, score) | pid7 out of range |",
              "|---|---:|---:|---:|"]
    for wave in WAVES:
        d = report["pid7_direction"][wave]
        c = report["party_match_corr"][wave]
        n = report["sentinels"][wave]["n_out_of_range"]
        lines.append(
            f"| {wave} | {'n/a' if d is None else f'{d:+.3f}'} | "
            f"{'n/a' if c is None else f'{c:+.3f}'} | {n:,} |"
        )

    lines += ["", "## Split composition", "", "| Split | rows by year |", "|---|---|"]
    for split, years in report["splits"].items():
        cells = ", ".join(f"{y}: {n:,}" for y, n in years.items())
        lines.append(f"| `{split}` | {cells} |")

    (out_dir / "00_covariate_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
