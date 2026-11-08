"""Build clean group-attitude splits.

Splits
------
train_t : 2016-2020
test_t  : 2022
train_r : 80% respondents (all years)
test_r  : 20% respondents (all years)

Usage:
    python -m experiments.data.build_group_splits
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from experiments import config
from experiments.data.prompts import build_group_prompt


def _is_test_respondent(rid) -> bool:
    h = hashlib.md5(str(int(float(rid))).encode()).hexdigest()
    return int(h, 16) % 10 < 2


PSEUDO_ID_BASE = 900_000


def _assign_block_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Assign block-level pseudo IDs to 2020 rows (which lost respondent_id).

    Every respondent block is a run of rows in which no `group_target` repeats,
    so a repeat marks the start of a new block. Block IDs are prefixed with
    `PSEUDO_ID_BASE` and are excluded from respondent-level splits.
    """
    df = df.copy()
    df["respondent_id"] = df["respondent_id"].astype(float)

    for year in df["year"].unique():
        mask = (df["year"] == year) & df["respondent_id"].isna()
        if not mask.any():
            continue
        ids = []
        block = PSEUDO_ID_BASE + int(year) * 1000
        seen: set[str] = set()
        for _, row in df.loc[mask].iterrows():
            group = str(row["group_target"])
            if group in seen:
                block += 1
                seen = set()
            seen.add(group)
            ids.append(block)
        df.loc[mask, "respondent_id"] = ids

    return df


def _write_jsonl(records: list[dict], path) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {
        "path": str(path),
        "n": len(records),
        "md5": hashlib.md5(path.read_bytes()).hexdigest(),
    }


def main() -> None:
    df = pd.read_csv(config.ANES_GROUPS_LONG, low_memory=False)
    meta = pd.read_csv(config.GROUP_METADATA)
    category = dict(zip(meta["group_name"], meta["category"]))

    df["feeling_thermometer"] = pd.to_numeric(df["feeling_thermometer"], errors="coerce")
    df = _assign_block_ids(df)

    before = len(df)
    df = df[df["feeling_thermometer"].between(0, 100)].copy()
    dropped_score = before - len(df)

    before = len(df)
    df = df.drop_duplicates(
        subset=["respondent_id", "year", "group_target"], keep="first"
    ).reset_index(drop=True)
    dropped_dup = before - len(df)

    df["_category"] = df["group_target"].map(category).fillna("unknown")

    def make_records(part: pd.DataFrame) -> list[dict]:
        out = []
        for _, row in part.iterrows():
            out.append({
                "instruction": build_group_prompt(row, row["_category"]),
                "input": "",
                "output": str(int(round(float(row["feeling_thermometer"])))),
                "metadata": {
                    "respondent_id": int(float(row["respondent_id"])),
                    "year": int(row["year"]),
                    "group": str(row["group_target"]),
                    "category": str(row["_category"]),
                    "true_score": float(row["feeling_thermometer"]),
                },
            })
        return out

    # Group coverage is year-dependent: only 2020 has all 9 groups, while
    # 2016/2018/2019/2022 have 3/6/4/2. Temporal splitting is therefore not
    # meaningful for groups. We provide two complementary protocols:
    #   1. 2020 block split: all 9 groups, unseen respondent blocks (80/20)
    #   2. multi-year respondent split: 6 groups with real IDs (80/20)
    df_2020 = df[df["year"] == 2020].copy()
    df_2020["_is_test"] = df_2020["respondent_id"].map(
        lambda x: int(hashlib.md5(str(int(x)).encode()).hexdigest(), 16) % 10 < 2
    )
    train_2020 = df_2020[~df_2020["_is_test"]].drop(columns="_is_test")
    test_2020 = df_2020[df_2020["_is_test"]].drop(columns="_is_test")

    real = df[df["respondent_id"] < PSEUDO_ID_BASE].copy()
    train_r = real[real["respondent_id"].map(lambda x: not _is_test_respondent(x))].copy()
    test_r = real[real["respondent_id"].map(_is_test_respondent)].copy()

    splits = {
        "train_2020": train_2020,
        "test_2020": test_2020,
        "train_r": train_r,
        "test_r": test_r,
    }

    config.V3_GROUPS.mkdir(parents=True, exist_ok=True)
    manifest = {"splits": {}, "cleaning": {
        "dropped_bad_score": int(dropped_score),
        "dropped_duplicates": int(dropped_dup),
        "pseudo_id_rows_2020": int((df["respondent_id"] >= PSEUDO_ID_BASE).sum()),
    }}
    for name, part in splits.items():
        manifest["splits"][name] = _write_jsonl(
            make_records(part), config.V3_GROUPS / f"{name}.jsonl"
        )
        manifest["splits"][name].update({
            "years": {int(k): int(v) for k, v in part["year"].value_counts().sort_index().items()},
            "n_respondents": int(part["respondent_id"].nunique()),
            "n_groups": int(part["group_target"].nunique()),
        })

    tr_resp = set(train_r["respondent_id"])
    te_resp = set(test_r["respondent_id"])
    assert not (tr_resp & te_resp)
    assert set(df["group_target"].unique()) == {
        "Black Americans", "White Americans", "Hispanic Americans", "Asian Americans",
        "Socialists", "Capitalists", "Feminists", "Democratic Party", "Republican Party",
    }
    manifest["checks"] = {
        "train_r_test_r_respondent_overlap": len(tr_resp & te_resp),
        "n_groups": int(df["group_target"].nunique()),
    }
    with open(config.V3_GROUPS / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
