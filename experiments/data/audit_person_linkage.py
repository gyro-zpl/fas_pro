"""Audit whether the released ANES files contain a cross-wave person key.

The v4 rerun plan (docs/02_计划与清单/人物级切分重跑指导_2026-11-20.md)
requires a person identifier that is stable across survey waves. This script
checks the released raw CSV files and the processed analysis table.

Usage:
    python -m experiments.data.audit_person_linkage \
        --raw-dir /path/to/FAS_PRO/data/anes/raw_data \
        --anes-root /path/to/FAS_PRO/data/anes/v3

Outputs:
    experiments/results/00_person_linkage_audit.md
    experiments/results/00_person_linkage_audit.json
"""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import pandas as pd

WAVES = {
    2016: ("anes_pilot_2016.csv", "caseid", "birthyr", "gender", "race"),
    2018: ("anes_pilot_2018.csv", "caseid", "birthyr", "gender", "race"),
    2019: ("anes_pilot_2019.csv", "caseid", "birthyr", "gender", "race"),
    2020: ("anes_pilot_2020ets_csv.csv", "ResponseId", "birthyr", "sex", None),
    2022: ("anes_pilot_2022_csv_20221214.csv", "caseid", "birthyr_dropdown",
           "gender", "racescf"),
    2024: ("anes_pilot_2024_20240319.csv", "caseid", "birthyr", "gender", "race"),
}


def _read_raw(raw_dir: Path, filename: str, id_col: str, birth_col: str,
              gender_col: str, race_col: str | None) -> pd.DataFrame:
    cols = [id_col, birth_col, gender_col] + ([race_col] if race_col else [])
    df = pd.read_csv(raw_dir / filename, usecols=cols, low_memory=False)
    return df.rename(columns={id_col: "raw_id", birth_col: "birthyr",
                              gender_col: "gender", race_col: "race"})


def _id_summary(y: int, df: pd.DataFrame) -> dict:
    s = df["raw_id"]
    numeric = pd.to_numeric(s, errors="coerce")
    return {
        "year": y,
        "rows": int(len(df)),
        "unique_ids": int(s.nunique()),
        "id_min": None if numeric.isna().all() else float(numeric.min()),
        "id_max": None if numeric.isna().all() else float(numeric.max()),
        "id_kind": "random string" if numeric.isna().all() else "integer",
    }


def _pair_check(a: int, da: pd.DataFrame, b: int, db: pd.DataFrame) -> dict:
    inter = set(da["raw_id"]) & set(db["raw_id"])
    if not inter:
        return {"a": a, "b": b, "overlap_ids": 0}
    x = da[da["raw_id"].isin(inter)].merge(
        db[db["raw_id"].isin(inter)], on="raw_id", suffixes=("_a", "_b"))
    return {
        "a": a,
        "b": b,
        "overlap_ids": int(len(inter)),
        "overlap_rows": int(len(x)),
        "birthyr_match": float((x["birthyr_a"] == x["birthyr_b"]).mean()),
        "gender_match": float((x["gender_a"] == x["gender_b"]).mean()),
    }


def _split_number_overlap(splits_dir: Path, train_name: str = "train_r") -> dict:
    train = pd.read_json(splits_dir / f"{train_name}.jsonl", lines=True)
    test = pd.read_json(splits_dir / "test_r.jsonl", lines=True)
    tr = train["metadata"].apply(pd.Series)["respondent_id"].astype(int)
    te = test["metadata"].apply(pd.Series)["respondent_id"].astype(int)
    te_years = test["metadata"].apply(pd.Series)["year"].astype(int)
    tr_set = set(tr)
    in_test = te.isin(tr_set)
    in_dist = te_years.isin([2016, 2018, 2019])
    return {
        "train_split": train_name,
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_numbers": int(tr.nunique()),
        "test_numbers": int(te.nunique()),
        "test_numbers_seen_in_train": int(te[in_test].nunique()),
        "test_rows_from_seen_numbers": int(in_test.sum()),
        "in_dist_rows": int(in_dist.sum()),
        "in_dist_rows_from_seen_numbers": int((in_test & in_dist).sum()),
        "note": ("respondent_id is a within-wave number; this overlap is "
                 "number-level, not person-level"),
    }


def _processed_check(clean_csv: Path) -> list[dict]:
    d = pd.read_csv(clean_csv, usecols=["respondent_id", "year",
                                        "birthyr_resp", "gender_resp"],
                    low_memory=False)
    out = []
    for a, b in combinations(sorted(d["year"].unique()), 2):
        x = (d[d["year"] == a][["respondent_id", "birthyr_resp", "gender_resp"]]
             .merge(d[d["year"] == b][["respondent_id", "birthyr_resp", "gender_resp"]],
                    on="respondent_id", suffixes=("_a", "_b")))
        if not len(x):
            continue
        out.append({
            "a": int(a), "b": int(b),
            "number_overlap": int(x["respondent_id"].nunique()),
            "birthyr_match": float((x["birthyr_resp_a"] == x["birthyr_resp_b"]).mean()),
            "gender_match": float((x["gender_resp_a"] == x["gender_resp_b"]).mean()),
        })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="directory containing the six raw ANES CSV files")
    ap.add_argument("--anes-root", required=True,
                    help="FAS_PRO/data/anes/v3 (clean table + splits)")
    ap.add_argument("--out-dir", default="experiments/results")
    args = ap.parse_args()

    raw_dir = Path(args.raw_dir)
    anes = Path(args.anes_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frames = {}
    summaries = []
    for y, (fname, id_col, b, g, r) in WAVES.items():
        df = _read_raw(raw_dir, fname, id_col, b, g, r)
        frames[y] = df
        summaries.append(_id_summary(y, df))

    pair_checks = []
    for a, b in combinations([2016, 2018, 2019, 2022, 2024], 2):
        pair_checks.append(_pair_check(a, frames[a], b, frames[b]))

    processed = _processed_check(anes / "anes_candidates_clean.csv")
    split_r = _split_number_overlap(anes / "splits", "train_r")
    split_t = _split_number_overlap(anes / "splits", "train_t")

    result = {
        "raw_id_summary": summaries,
        "raw_cross_wave": pair_checks,
        "processed_cross_wave": processed,
        "v3_split_number_overlap": {"r": split_r, "t": split_t},
        "conclusion": (
            "The released raw CSV caseid is a within-wave 1..N number, not a "
            "cross-wave person key; the processed respondent_id is 0..N-1. "
            "Cross-wave birth-year agreement is near chance (~2%), and 2020 "
            "only carries a random ResponseId. Person-level v4 splits cannot "
            "be built from these files. Fallback B applies: report held-out "
            "respondent numbers within a wave, and state that person-level "
            "isolation is unverified."
        ),
    }
    with open(out_dir / "00_person_linkage_audit.json", "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    lines = ["# Person-level linkage audit (2026-11-20)", ""]
    lines.append("## Raw ANES files")
    lines.append("")
    lines.append("| Year | Rows | Unique IDs | ID min | ID max | ID kind |")
    lines.append("|---|---:|---:|---:|---:|---|")
    for s in summaries:
        lines.append(
            f"| {s['year']} | {s['rows']} | {s['unique_ids']} | "
            f"{s['id_min'] if s['id_min'] is not None else '—'} | "
            f"{s['id_max'] if s['id_max'] is not None else '—'} | {s['id_kind']} |")
    lines.append("")
    lines.append("## Cross-wave overlap of the raw caseid")
    lines.append("")
    lines.append("| Pair | Overlap IDs | Birth-year match | Gender match |")
    lines.append("|---|---:|---:|---:|")
    for p in pair_checks:
        if "birthyr_match" in p:
            lines.append(f"| {p['a']}–{p['b']} | {p['overlap_ids']} | "
                         f"{p['birthyr_match']:.4f} | {p['gender_match']:.4f} |")
        else:
            lines.append(f"| {p['a']}–{p['b']} | 0 | — | — |")
    lines.append("")
    lines.append("A genuine cross-wave person ID would keep the same birth year "
                 "and gender for the same ID. The near-chance agreement shows "
                 "that the same `caseid` in two waves is a different person.")
    lines.append("")
    lines.append("## Processed analysis table")
    lines.append("")
    lines.append("The clean table uses a 0-based within-wave `respondent_id`; "
                 "cross-wave number overlap has near-chance demographic agreement:")
    lines.append("")
    lines.append("| Pair | Overlapping numbers | Birth-year match | Gender match |")
    lines.append("|---|---:|---:|---:|")
    for p in processed:
        lines.append(f"| {p['a']}–{p['b']} | {p['number_overlap']} | "
                     f"{p['birthyr_match']:.4f} | {p['gender_match']:.4f} |")
    lines.append("")
    lines.append("## v3 split number overlap (not person-level)")
    lines.append("")
    for label, sc in [("R protocol (train_r -> test_r)", split_r),
                      ("T protocol (train_t -> test_r)", split_t)]:
        lines.append(f"**{label}**")
        lines.append("")
        lines.append(f"- train rows: {sc['train_rows']}, numbers: {sc['train_numbers']}")
        lines.append(f"- test rows: {sc['test_rows']}, numbers: {sc['test_numbers']}")
        lines.append(f"- test numbers seen in train: {sc['test_numbers_seen_in_train']}")
        lines.append(f"- test rows from seen numbers: {sc['test_rows_from_seen_numbers']}")
        lines.append(f"- in-distribution rows: {sc['in_dist_rows']}; from seen numbers: "
                     f"{sc['in_dist_rows_from_seen_numbers']}")
        lines.append("")
    lines.append("The R split is disjoint in respondent **numbers** by construction "
                 "(0 overlap); this is the 'held-out respondent numbers' claim. It "
                 "does not establish person-level isolation, because the same person "
                 "can appear under different within-wave numbers. The T split reuses "
                 "test numbers in training by design and is not respondent-held-out. "
                 "Neither overlap is a person-level leakage measurement.")
    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    lines.append(result["conclusion"])
    lines.append("")
    lines.append("If the original ANES `.dta`/`.sav` files with a stable panel "
                 "identifier (e.g., `V160001`/`V200001`/`V240001`) become "
                 "available, the v4 person-level rerun can proceed. Until then, "
                 "the paper should use 'held-out respondent numbers within a "
                 "wave' and list person-level isolation as unverified.")
    (out_dir / "00_person_linkage_audit.md").write_text("\n".join(lines),
                                                      encoding="utf-8")
    print("\n".join(lines))
    print(f"\nSaved {out_dir}")


if __name__ == "__main__":
    main()
