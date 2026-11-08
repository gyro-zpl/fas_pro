"""Clean the candidate table.

Actions:
1. keep original row index as `orig_row_id` (needed by the tabular baseline)
2. normalize candidate names (`Obama.` -> `obama`)
3. drop social-group targets that were mixed into the candidate table
4. drop missing / out-of-range support scores
5. deduplicate `(respondent_id, year, candidate)` keeping the first row

Usage:
    python -m experiments.data.clean_candidates
Outputs:
    data/anes/v3/anes_candidates_clean.csv
    data/anes/v3/clean_manifest.json
"""
from __future__ import annotations

import json

import pandas as pd

from experiments import config


def _norm_name(s: str) -> str:
    return str(s).strip().lower().rstrip(".")


def main() -> None:
    df = pd.read_csv(config.ANES_FEATURES, low_memory=False)
    df["orig_row_id"] = range(len(df))
    total = len(df)

    df["candidate_raw"] = df["candidate"]
    df["candidate"] = df["candidate"].map(_norm_name)

    group_mask = df["candidate"].isin(config.GROUP_LIKE_TARGETS)
    n_group = int(group_mask.sum())
    df = df.loc[~group_mask].copy()

    score = pd.to_numeric(df["support_score"], errors="coerce")
    bad_score = score.isna() | (score < 0) | (score > 100)
    n_bad = int(bad_score.sum())
    df = df.loc[~bad_score].copy()
    df["support_score"] = score.loc[~bad_score].astype(float)

    # Record duplicate score disagreement before dropping.
    key = ["respondent_id", "year", "candidate"]
    dup = df[df.duplicated(subset=key, keep=False)]
    disagreement = 0
    if len(dup):
        spread = dup.groupby(key)["support_score"].nunique()
        disagreement = int((spread > 1).sum())
    before = len(df)
    df = df.drop_duplicates(subset=key, keep="first").reset_index(drop=True)
    n_dropped_dup = before - len(df)

    keep_cols = [c for c in df.columns if c not in {"candidate_raw"}]
    df = df[keep_cols]

    config.V3.mkdir(parents=True, exist_ok=True)
    out_csv = config.V3 / "anes_candidates_clean.csv"
    df.to_csv(out_csv, index=False)

    manifest = {
        "source": str(config.ANES_FEATURES),
        "output": str(out_csv),
        "rows_before": total,
        "rows_dropped_group_targets": n_group,
        "rows_dropped_bad_score": n_bad,
        "rows_dropped_duplicates": n_dropped_dup,
        "duplicate_groups_with_disagreeing_scores": disagreement,
        "rows_after": len(df),
        "respondents_after": int(df["respondent_id"].nunique()),
        "years_after": {int(y): int(c) for y, c in df["year"].value_counts().sort_index().items()},
        "candidates_after": {
            int(y): sorted(g["candidate"].unique().tolist())
            for y, g in df.groupby("year")
        },
    }
    with open(config.V3 / "clean_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps({k: v for k, v in manifest.items() if k not in {"candidates_after"}}, indent=2))
    print("candidates per year:")
    for y, cs in manifest["candidates_after"].items():
        print(f"  {y}: {len(cs)} -> {cs}")


if __name__ == "__main__":
    main()
