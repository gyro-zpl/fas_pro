"""Audit the raw ANES feature table before cleaning.

Usage:
    python -m experiments.data.audit_raw
Outputs:
    experiments/results/00_audit.md
    experiments/results/00_audit.json
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

from experiments import config


def main() -> None:
    df = pd.read_csv(config.ANES_FEATURES, low_memory=False)
    n = len(df)

    # --- basic ---
    year_counts = df["year"].value_counts().sort_index().to_dict()
    candidates = df["candidate"].astype(str)
    respondents = df["respondent_id"].astype(str)

    # --- abnormal candidate names ---
    trailing_dot = candidates.str.endswith(".").sum()
    whitespace = (candidates != candidates.str.strip()).sum()
    upper = candidates.str.contains(r"[A-Z]").sum()
    group_like = candidates.str.lower().isin(config.GROUP_LIKE_TARGETS).sum()

    # --- target sanity ---
    score = pd.to_numeric(df["support_score"], errors="coerce")
    missing = int(score.isna().sum())
    out_of_range = int(((score < 0) | (score > 100)).sum())

    # --- panel structure ---
    years_per_resp = df.groupby("respondent_id")["year"].nunique()
    train_years = set(config.TRAIN_YEARS)
    test_years = set(config.TEST_YEARS) | set(config.VAL_YEARS)
    tr = set(df.loc[df.year.isin(train_years), "respondent_id"])
    te = set(df.loc[df.year.isin(test_years), "respondent_id"])
    overlap = len(tr & te)

    # --- duplicates ---
    dup_keys = df.duplicated(subset=["respondent_id", "year", "candidate"], keep=False)
    dup_rows = int(dup_keys.sum())
    dup_groups = int(
        df.loc[dup_keys].groupby(["respondent_id", "year", "candidate"]).ngroups
    ) if dup_rows else 0

    # --- candidate sets per year (raw vs after name normalization) ---
    def norm_name(s: str) -> str:
        return str(s).strip().lower().rstrip(".")

    per_year_raw = {
        int(y): sorted(g["candidate"].astype(str).unique().tolist())
        for y, g in df.groupby("year")
    }
    tmp = df.copy()
    tmp["candidate_norm"] = tmp["candidate"].map(norm_name)
    tmp = tmp[~tmp["candidate_norm"].isin(config.GROUP_LIKE_TARGETS)]
    per_year_clean = {
        int(y): sorted(g["candidate_norm"].unique().tolist())
        for y, g in tmp.groupby("year")
    }

    report = {
        "n_rows": n,
        "year_counts": {int(k): int(v) for k, v in year_counts.items()},
        "n_respondents": int(df["respondent_id"].nunique()),
        "respondents_in_multiple_years": int((years_per_resp > 1).sum()),
        "max_years_per_respondent": int(years_per_resp.max()),
        "train_test_respondent_overlap": overlap,
        "test_only_respondents": len(te - tr),
        "candidate_name_issues": {
            "trailing_dot_rows": int(trailing_dot),
            "whitespace_rows": int(whitespace),
            "uppercase_rows": int(upper),
            "group_like_rows": int(group_like),
        },
        "target_issues": {
            "missing": missing,
            "out_of_range": out_of_range,
        },
        "duplicate_rows_same_key": dup_rows,
        "duplicate_keys": dup_groups,
        "candidates_per_year_raw": per_year_raw,
        "candidates_per_year_clean": per_year_clean,
    }

    config.RESULTS.mkdir(parents=True, exist_ok=True)
    with open(config.RESULTS / "00_audit.json", "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        "# Raw ANES audit",
        "",
        f"- rows: **{n:,}**",
        f"- respondents: **{report['n_respondents']:,}**",
        f"- respondents in >1 year: **{report['respondents_in_multiple_years']:,}**",
        f"- max years per respondent: **{report['max_years_per_respondent']}**",
        f"- train-year vs test-year respondent overlap: **{overlap:,}**",
        f"- test-only respondents: **{report['test_only_respondents']}**",
        "",
        "## Candidate name issues",
        "",
        f"- trailing dot rows: **{trailing_dot:,}**",
        f"- whitespace rows: **{whitespace:,}**",
        f"- uppercase rows: **{upper:,}**",
        f"- group-like targets mixed in: **{group_like:,}**",
        "",
        "## Target",
        "",
        f"- missing `support_score`: **{missing:,}**",
        f"- out of range: **{out_of_range:,}**",
        "",
        "## Duplicates",
        "",
        f"- rows sharing `(respondent_id, year, candidate)`: **{dup_rows:,}**",
        f"- duplicate groups: **{dup_groups:,}**",
        "",
        "## Rows per year",
        "",
        "| year | rows |",
        "|---|---|",
    ]
    for y, c in report["year_counts"].items():
        lines.append(f"| {y} | {c:,} |")
    lines += ["", "## Candidates per year (raw)", ""]
    for y, cs in per_year_raw.items():
        lines.append(f"- **{y}** ({len(cs)}): {', '.join(cs)}")
    lines += ["", "## Candidates per year (normalized, groups removed)", ""]
    for y, cs in per_year_clean.items():
        lines.append(f"- **{y}** ({len(cs)}): {', '.join(cs)}")

    (config.RESULTS / "00_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines[:30]))
    print(f"\nSaved to {config.RESULTS / '00_audit.md'}")


if __name__ == "__main__":
    main()
