"""Build clean candidate splits.

Splits
------
train_t : sampled training pool, years 2016-2020
val_t   : all 2022 rows (capped by respondents)
test_t  : all 2024 rows
train_r : sampled training pool, 80% respondents
test_r  : 20% respondents, sampled across years
test_tr : 2024 rows of the 20% respondents

The training pool comes from the existing sampled file (`train_dataset_fixed.jsonl`)
so we keep the prior sampling design. Test splits are built from the cleaned full
table, sampling by respondent so candidate sets stay complete.

Usage:
    python -m experiments.data.build_splits
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments import config
from experiments.data.prompts import build_candidate_prompt


def _is_test_respondent(rid) -> bool:
    h = hashlib.md5(str(int(float(rid))).encode()).hexdigest()
    return int(h, 16) % 10 < 2


def _sample_by_respondent(df: pd.DataFrame, max_rows: int, seed: int = 42) -> pd.DataFrame:
    """Take whole respondents until `max_rows` is reached."""
    if len(df) <= max_rows:
        return df
    rng = pd.Series(df["respondent_id"].unique()).sample(frac=1.0, random_state=seed)
    counts = df.groupby("respondent_id").size()
    chosen, total = [], 0
    for rid in rng:
        chosen.append(rid)
        total += int(counts.loc[rid])
        if total >= max_rows:
            break
    return df[df["respondent_id"].isin(chosen)]


def _load_train_row_ids(path: Path) -> dict[int, dict]:
    mapping: dict[int, dict] = {}
    if not path.exists():
        return mapping
    with open(path) as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            meta = obj.get("metadata") or {}
            rid = meta.get("row_id")
            if rid is not None:
                mapping[int(rid)] = meta
    return mapping


def _to_records(df: pd.DataFrame, strategy_default: str = "clean") -> list[dict]:
    records = []
    for _, row in df.iterrows():
        instruction = build_candidate_prompt(row, level=config.PROMPT_LEVEL)
        records.append({
            "instruction": instruction,
            "input": "",
            "output": str(int(round(float(row["support_score"])))),
            "metadata": {
                "respondent_id": int(float(row["respondent_id"])),
                "year": int(row["year"]),
                "candidate": str(row["candidate"]),
                "true_score": float(row["support_score"]),
                "strategy": row.get("_strategy", strategy_default),
                "party_match": None if pd.isna(row.get("party_match")) else int(row.get("party_match")),
                "ideology_distance_7": None if pd.isna(row.get("ideology_distance_7")) else float(row.get("ideology_distance_7")),
                "orig_row_id": int(row["orig_row_id"]),
            },
        })
    return records


def _write_jsonl(records: list[dict], path: Path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    h = hashlib.md5(path.read_bytes()).hexdigest()
    return {"path": str(path), "n": len(records), "md5": h}


def main() -> None:
    df = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    df["respondent_id"] = df["respondent_id"].astype(float)

    # Attach the original sampling strategy where available.
    row_meta = _load_train_row_ids(config.TRAIN_SOURCE)
    df["_strategy"] = df["orig_row_id"].map(
        lambda x: (row_meta.get(int(x)) or {}).get("strategy", "clean")
    )

    # Training pool: rows that appeared in the sampled training file.
    pool = df[df["orig_row_id"].isin(row_meta.keys())].copy()
    if pool.empty:
        raise RuntimeError(f"No sampled training rows matched: {config.TRAIN_SOURCE}")
    matched = len(pool)
    unmatched = len(row_meta) - matched

    train_t = pool[pool["year"].isin(config.TRAIN_YEARS)].copy()
    train_r = train_t[train_t["respondent_id"].map(lambda x: not _is_test_respondent(x))].copy()

    val_t = df[df["year"].isin(config.VAL_YEARS)].copy()
    val_t = _sample_by_respondent(val_t, max_rows=2500)

    test_t = df[df["year"].isin(config.TEST_YEARS)].copy()

    test_r_full = df[df["respondent_id"].map(_is_test_respondent)].copy()
    test_r = _sample_by_respondent(test_r_full, max_rows=6000)

    # Keep test_tr a strict subset of test_r so the two reported numbers come
    # from the same respondents.
    test_tr = test_r[test_r["year"].isin(config.TEST_YEARS)].copy()

    splits = {
        "train_t": train_t,
        "val_t": val_t,
        "test_t": test_t,
        "train_r": train_r,
        "test_r": test_r,
        "test_tr": test_tr,
    }

    config.V3_SPLITS.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}, "train_source": str(config.TRAIN_SOURCE)}
    for name, part in splits.items():
        records = _to_records(part)
        manifest["splits"][name] = _write_jsonl(records, config.V3_SPLITS / f"{name}.jsonl")
        manifest["splits"][name].update({
            "years": {int(k): int(v) for k, v in part["year"].value_counts().sort_index().items()},
            "n_respondents": int(part["respondent_id"].nunique()),
            "n_candidates": int(part["candidate"].nunique()),
        })

    # --- assertions ---
    train_years = set(train_t["year"].unique())
    test_years = set(test_t["year"].unique())
    assert not (train_years & test_years), (train_years, test_years)

    tr_resp = set(train_r["respondent_id"])
    te_resp = set(test_r["respondent_id"])
    assert not (tr_resp & te_resp), "respondent leakage between train_r and test_r"

    # Row-level overlap. The two protocols are alternative designs, so we only
    # require disjointness *within* each protocol, not across all splits.
    row_sets = {
        name: set(part["orig_row_id"].tolist()) for name, part in splits.items()
    }

    def overlap(a: str, b: str) -> int:
        return len(row_sets[a] & row_sets[b])

    within_protocol = {
        "train_t__val_t": overlap("train_t", "val_t"),
        "train_t__test_t": overlap("train_t", "test_t"),
        "train_r__test_r": overlap("train_r", "test_r"),
        "train_r__test_tr": overlap("train_r", "test_tr"),
    }
    for pair, n in within_protocol.items():
        assert n == 0, f"leak within protocol: {pair} shares {n} rows"

    # Prompt collisions are a data property, not leakage: different respondents
    # can share every visible attribute but have different scores.
    prompt_targets: dict[str, set] = {}
    for name in splits:
        with open(config.V3_SPLITS / f"{name}.jsonl") as f:
            for line in f:
                obj = json.loads(line)
                h = hashlib.md5(obj["instruction"].encode()).hexdigest()
                prompt_targets.setdefault(h, set()).add(float(obj["metadata"]["true_score"]))
    n_collisions = sum(1 for v in prompt_targets.values() if len(v) > 1)

    assert set(test_t["candidate"].unique()) == {"biden", "trump"}, \
        set(test_t["candidate"].unique())

    manifest["checks"] = {
        "train_test_year_overlap": sorted(train_years & test_years),
        "train_r_test_r_respondent_overlap": len(tr_resp & te_resp),
        "within_protocol_row_overlap": within_protocol,
        "cross_protocol_test_t__test_r_rows": overlap("test_t", "test_r"),
        "test_tr_is_subset_of_test_r": row_sets["test_tr"].issubset(row_sets["test_r"]),
        "prompts_with_conflicting_targets": n_collisions,
        "test_t_candidates": sorted(test_t["candidate"].unique().tolist()),
        "sampled_rows_matched": matched,
        "sampled_rows_unmatched_after_cleaning": unmatched,
    }
    with open(config.V3_SPLITS / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest["splits"], indent=2, ensure_ascii=False))
    print(json.dumps(manifest["checks"], indent=2, ensure_ascii=False))
    print(f"\nSaved to {config.V3_SPLITS}")


if __name__ == "__main__":
    main()
