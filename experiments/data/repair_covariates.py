"""Repair the ANES political covariates in a separate v4 table.

Read-only with respect to ``data/anes/v3``. The v3 analysis table loses the
2024 party/ideology/education items because the 2024 extract stores them as
text labels, uses the placeholder 2018 ``pid7`` column instead of ``pid7x``,
and keeps the reversed 2020 ``pid7`` scale. This script reconstructs the
covariates from the raw ANES CSVs and writes:

    data/anes/v4_covariates/anes_covariates_repaired.csv
    data/anes/v4_covariates/anes_candidates_base_v4cov.csv
    data/anes/v4_covariates/README.md
    experiments/results/00_covariate_repair.md/.json

The row-level output contains only base columns and the two prompt-visible
derived columns that must change with the covariates (``party_match`` and
``ideology_distance_7``). The remaining interaction features are intentionally
not recomputed here; that is the next step before any retraining.

Usage:
    python -m experiments.data.repair_covariates \
        --raw-dir /path/to/FAS_PRO/data/anes/raw_data
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import config
from experiments.data import prompts

DEFAULT_RAW = Path("/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/raw_data")
WAVES = [2016, 2018, 2019, 2020, 2022, 2024]
COVARIATES = ["pid7", "ideo5", "educ", "lcself"]

PID7_TEXT = {
    "Strong Democrat": 1,
    "Not very strong Democrat": 2,
    "Lean Democrat": 3,
    "Independent": 4,
    "Lean Republican": 5,
    "Not very strong Republican": 6,
    "Strong Republican": 7,
}
IDEO5_TEXT = {
    "Very liberal": 1,
    "Liberal": 2,
    "Moderate": 3,
    "Conservative": 4,
    "Very conservative": 5,
}
EDUC_TEXT = {
    "No HS credential": 1,
    "High school graduate": 3,
    "Some college": 4,
    "2-year degree": 4,
    "4-year degree": 5,
    "Post-grad": 6,
}
LCSELF_TEXT = {
    "Extremely liberal": 1,
    "Liberal": 2,
    "Slightly liberal": 3,
    "Moderate; middle of the road": 4,
    "Slightly conservative": 5,
    "Conservative": 6,
    "Extremely conservative": 7,
}


def _clean_pid7(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(1, 7))


def _clean_ideo5(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(1, 5))


def _clean_educ(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(1, 6))


def _clean_lcself(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.where(s.between(1, 7))


def _lcself_to_ideo5(s: pd.Series) -> pd.Series:
    """Map the 7-point liberal-conservative scale to the 5-point item."""
    mapping = {1: 1, 2: 2, 3: 2, 4: 3, 5: 4, 6: 4, 7: 5}
    return s.map(mapping)


def _raw_wave_map(raw: Path, year: int, v3: pd.DataFrame) -> pd.DataFrame:
    """Return year, respondent_id, repaired pid7/ideo5/educ/lcself."""
    if year == 2018:
        wide = pd.read_csv(raw / "anes_pilot_2018.csv", low_memory=False)
        wide["respondent_id"] = np.arange(len(wide))
        long = pd.read_csv(raw / "anes_pilot_2018_long.csv", low_memory=False)
        base = long[["respondent_id", "ideo5", "educ", "lcself"]].drop_duplicates(
            "respondent_id"
        )
        base = base.merge(wide[["respondent_id", "pid7x"]], on="respondent_id")
        base = base.rename(columns={"pid7x": "pid7"})
        base["pid7"] = _clean_pid7(base["pid7"])
        base["ideo5"] = _clean_ideo5(base["ideo5"])
        base["educ"] = _clean_educ(base["educ"])
        base["lcself"] = _clean_lcself(base["lcself"])
    elif year == 2020:
        long = pd.read_csv(raw / "anes_pilot_2020_long.csv", low_memory=False)
        base = long[["respondent_id", "pid7", "educ", "lcself"]].drop_duplicates(
            "respondent_id"
        )
        base["pid7"] = _clean_pid7(base["pid7"])
        # The 2020 pid7 scale runs in the opposite direction.
        base.loc[base["pid7"].notna(), "pid7"] = 8 - base.loc[
            base["pid7"].notna(), "pid7"
        ]
        base["lcself"] = _clean_lcself(base["lcself"])
        base["ideo5"] = _lcself_to_ideo5(base["lcself"])
        base["educ"] = _clean_educ(base["educ"])
    elif year == 2024:
        wide = pd.read_csv(raw / "anes_pilot_2024_20240319.csv", low_memory=False)
        wide["respondent_id"] = np.arange(len(wide))
        long = pd.read_csv(raw / "anes_pilot_2024_long.csv", low_memory=False)
        base = long[["respondent_id", "pid7", "ideo5", "educ"]].drop_duplicates(
            "respondent_id"
        )
        base = base.merge(
            wide[["respondent_id", "ideology_lcself"]], on="respondent_id"
        )
        base["pid7"] = base["pid7"].map(PID7_TEXT)
        base["ideo5"] = base["ideo5"].map(IDEO5_TEXT)
        base["educ"] = base["educ"].map(EDUC_TEXT)
        base["lcself"] = base["ideology_lcself"].map(LCSELF_TEXT)
        base = base.drop(columns=["ideology_lcself"])
    else:
        # 2016/2019/2022 use the cleaned values, restricted to the valid range.
        base = (
            v3.loc[v3["year"] == year, ["respondent_id", *COVARIATES]]
            .drop_duplicates("respondent_id")
            .copy()
        )
        base["pid7"] = _clean_pid7(base["pid7"])
        base["ideo5"] = _clean_ideo5(base["ideo5"])
        base["educ"] = _clean_educ(base["educ"])
        base["lcself"] = _clean_lcself(base["lcself"])

    base.insert(0, "year", year)
    base["respondent_id"] = base["respondent_id"].astype(int)
    return base


def _party_match(df: pd.DataFrame) -> pd.Series:
    align = pd.Series(0, index=df.index, dtype=int)
    align.loc[df["pid7"] <= 3] = 1
    align.loc[df["pid7"] >= 5] = 2
    match = (align == df["party"]).astype(int)
    match.loc[align == 0] = -1
    match.loc[df["pid7"].isna()] = -1
    return match


def _profile_count(df: pd.DataFrame) -> int:
    profiles = set()
    for _, row in df.iterrows():
        text = prompts.build_candidate_prompt(row, level=config.PROMPT_LEVEL)
        block = " | ".join(
            line
            for line in text.splitlines()
            if line.startswith(
                ("- Education:", "- Political:", "- Party alignment:",
                 "- Ideology distance:")
            )
        )
        profiles.add(block)
    return len(profiles)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=str(DEFAULT_RAW))
    p.add_argument("--out-dir", default=str(config.ROOT / "data/anes/v4_covariates"))
    p.add_argument("--report-dir", default=str(config.RESULTS))
    args = p.parse_args()
    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    v3 = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    v3["pid7"] = pd.to_numeric(v3["pid7"], errors="coerce")
    v3["ideo5"] = pd.to_numeric(v3["ideo5"], errors="coerce")
    v3["educ"] = pd.to_numeric(v3["educ"], errors="coerce")
    v3["lcself"] = pd.to_numeric(v3["lcself"], errors="coerce")
    v3["party_match"] = pd.to_numeric(v3["party_match"], errors="coerce")

    repaired = pd.concat(
        [_raw_wave_map(raw, year, v3) for year in WAVES], ignore_index=True
    )
    repaired.to_csv(out / "anes_covariates_repaired.csv", index=False)

    # Row-level base table: v3 columns used by the prompt builder, with the
    # repaired covariates and the two prompt-visible derived columns.
    base_cols = [
        "orig_row_id", "respondent_id", "year", "candidate", "support_score",
        "is_missing", "birthyr_resp", "gender_resp", "race_resp", "educ",
        "marstat", "employ", "faminc", "votereg", "pid7", "ideo5", "lcself",
        "age_resp", "age_cand", "age_difference", "party", "position_encoded",
        "ideology_cand",
    ]
    base = v3[base_cols].drop(columns=COVARIATES).copy()
    base = base.merge(
        repaired[["year", "respondent_id", *COVARIATES]],
        on=["year", "respondent_id"],
        how="left",
    )
    base["party_match"] = _party_match(base)
    base["ideology_distance_7"] = (base["lcself"] - base["ideology_cand"]).abs()
    base.to_csv(out / "anes_candidates_base_v4cov.csv", index=False)

    report = {"mapping": {}, "coverage": {}, "profiles": {}, "direction": {}}
    for year in WAVES:
        before = v3[v3["year"] == year].copy()
        after = base[base["year"] == year].copy()
        # The profile comparison uses the same rows; the base table is aligned
        # with v3 by orig_row_id, so the order is preserved within each wave.
        report["coverage"][str(year)] = {
            "rows": int(len(before)),
            "before": {
                c: round(float(before[c].notna().mean()), 4) for c in COVARIATES
            },
            "after": {
                c: round(float(after[c].notna().mean()), 4) for c in COVARIATES
            },
        }
        report["profiles"][str(year)] = {
            "before": _profile_count(before),
            "after": _profile_count(after),
        }
        ok = after[after["pid7"].between(1, 7) & after["lcself"].between(1, 7)]
        report["direction"][str(year)] = (
            round(float(ok["pid7"].corr(ok["lcself"])), 4) if len(ok) > 10 else None
        )

    report["mapping"] = {
        "2018": "raw pid7x instead of the placeholder pid7",
        "2020": "reverse pid7 (8 - pid7); ideo5 from lcself",
        "2024": "text pid7/ideo5/educ mapped to numeric; lcself from ideology_lcself",
        "2016/2019/2022": "v3 values restricted to valid ranges",
    }
    (report_dir / "00_covariate_repair.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Covariate repair audit",
        "",
        "Reconstructed from the raw ANES CSVs without modifying `data/anes/v3`.",
        "",
        "| Wave | Rows | pid7 before/after | ideo5 before/after | educ before/after | lcself before/after | Profiles before/after | corr(pid7, lcself) |",
        "|---|---:|---|---|---|---|---|---:|",
    ]
    for year in WAVES:
        c = report["coverage"][str(year)]
        pcount = report["profiles"][str(year)]
        def fmt(k):
            return f"{c['before'][k] * 100:.0f}% / {c['after'][k] * 100:.0f}%"
        d = report["direction"][str(year)]
        lines.append(
            f"| {year} | {c['rows']:,} | {fmt('pid7')} | {fmt('ideo5')} | "
            f"{fmt('educ')} | {fmt('lcself')} | {pcount['before']} / "
            f"{pcount['after']} | {'n/a' if d is None else f'{d:+.3f}'} |"
        )
    lines += [
        "",
        "Coverage is the share of rows with a substantive value after restricting",
        "each item to its valid range (pid7 1--7, ideo5 1--5, educ 1--6, lcself 1--7).",
        "`Profiles` counts distinct education/political/alignment/distance blocks",
        "rendered by `prompts.build_candidate_prompt`.",
        "",
        "## Mapping",
        "",
        "- 2018: use `pid7x`; the `pid7` column is a placeholder (`-4`).",
        "- 2020: `pid7` is reverse-coded; use `8 - pid7`. `ideo5` is not in the raw",
        "  file; derive it from `lcself`.",
        "- 2024: raw `pid7`, `ideo5`, and `educ` are text labels; map them to numeric",
        "  codes. `lcself` comes from `ideology_lcself`.",
        "- 2016/2019/2022: v3 values restricted to valid ranges.",
        "",
        "The row-level base table is `anes_candidates_base_v4cov.csv`; it is not a",
        "training table until the interaction features are recomputed.",
    ]
    (report_dir / "00_covariate_repair.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))
    print(f"\nwrote {out / 'anes_covariates_repaired.csv'}")
    print(f"wrote {out / 'anes_candidates_base_v4cov.csv'}")


if __name__ == "__main__":
    main()
