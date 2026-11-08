"""Build the three ablation datasets.

A1 random sampling : same size as train_t, but sampled uniformly from all
                     clean train-year rows (removes the 9-strategy design).
A2 minimal prompt  : same rows as train_t, but prompt drops interaction features.
A3 name-only       : same rows, but prompt drops candidate party/position/age
                     and interaction features.

Each ablation also gets a matching test_r variant so training and evaluation
prompts are consistent.

Usage:
    python -m experiments.data.build_ablation_splits
"""
from __future__ import annotations

import hashlib
import json

import pandas as pd

from experiments import config
from experiments.data.prompts import build_candidate_prompt

OUT = config.V3 / "ablations"
OUT.mkdir(parents=True, exist_ok=True)


def _load_ids(path) -> list[int]:
    ids = []
    with open(path) as f:
        for line in f:
            ids.append(int(json.loads(line)["metadata"]["orig_row_id"]))
    return ids


def _records(df: pd.DataFrame, level: str) -> list[dict]:
    out = []
    for _, row in df.iterrows():
        out.append({
            "instruction": build_candidate_prompt(row, level=level),
            "input": "",
            "output": str(int(round(float(row["support_score"])))),
            "metadata": {
                "respondent_id": int(float(row["respondent_id"])),
                "year": int(row["year"]),
                "candidate": str(row["candidate"]),
                "true_score": float(row["support_score"]),
                "orig_row_id": int(row["orig_row_id"]),
                "ablation": level,
            },
        })
    return out


def _write(records: list[dict], name: str) -> dict:
    path = OUT / name
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"path": str(path), "n": len(records),
            "md5": hashlib.md5(path.read_bytes()).hexdigest()}


def main() -> None:
    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean = clean.set_index("orig_row_id", drop=False)

    train_ids = _load_ids(config.V3_SPLITS / "train_t.jsonl")
    test_ids = _load_ids(config.V3_SPLITS / "test_r.jsonl")

    train_df = clean.loc[train_ids].reset_index(drop=True)
    test_df = clean.loc[test_ids].reset_index(drop=True)

    # A1: uniform random sample from all clean train-year rows, same size.
    pool = clean[clean["year"].isin(config.TRAIN_YEARS)]
    a1 = pool.sample(n=len(train_df), random_state=42).reset_index(drop=True)

    manifest = {
        "a1_random": {
            "train": _write(_records(a1, "core"), "train_a1_random.jsonl"),
        },
        "a2_minimal": {
            "train": _write(_records(train_df, "minimal"), "train_a2_minimal.jsonl"),
            "test_r": _write(_records(test_df, "minimal"), "test_r_a2_minimal.jsonl"),
        },
        "a3_name_only": {
            "train": _write(_records(train_df, "name_only"), "train_a3_name_only.jsonl"),
            "test_r": _write(_records(test_df, "name_only"), "test_r_a3_name_only.jsonl"),
        },
    }
    with open(OUT / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
