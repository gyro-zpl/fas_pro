"""Covariate-dropout probe for the 2024 temporal collapse.

Question: does removing the voter's political covariates from a 2019 prompt
reproduce the 2024 collapse, or does the collapse persist when those covariates
are present?

The probe is read-only with respect to training:

* LLM: reuse the committed seed-42 predictions for the original 2019 rows, then
  run the same LoRA adapter on the same rows with the political block replaced
  by the 2024 defaults (Independent, Moderate; no education; independent voter;
  ideology distance 0.0).
* HistGB: fit the existing tabular recipe on ``train_r`` once, then predict the
  original rows and a copy whose voter-political columns are set to missing
  (2024-style). Candidate metadata and demographics are retained.

Outputs:
    experiments/results/diagnostics/covariate_dropout_probe.md
    experiments/results/diagnostics/covariate_dropout_probe.json
    experiments/results/diagnostics/covariate_dropout_probe_ablated.predictions.jsonl

Usage:
    CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python -m \
        experiments.diagnostics.covariate_dropout_probe \
        --lora /path/to/qwen3_8b_r_seed42
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import config
from experiments.baselines.tabular import CAT_COLS, _design
from experiments.eval import metrics

DEFAULT_LORA = Path(
    "/home/jpzhao_team/cyyuan/FAS_PRO/sft/sft_models/qwen3_8b_r_seed42"
)
DEFAULT_PREDS = config.PREDS / "qwen3_r_seed42_test_r_generate.predictions.jsonl"

# Columns that carry the voter's party/ideology or are derived from them.
# Candidate metadata (party, position, candidate ideology) is deliberately kept.
POLITICAL_COLS = [
    "pid7", "ideo5", "lcself", "educ",
    "party_match", "party_alignment", "same_party", "opposite_party",
    "party_strength_resp", "party_distance", "party_strength_x_match",
    "party_strength_x_distance", "ideology_distance_5", "ideology_same_side_5",
    "ideology_distance_7", "ideology_same_side_7", "both_extreme_ideology",
    "both_moderate_ideology", "ideology_strength_resp",
    "party_match_x_ideology_dist", "party_same_x_ideology_same",
    "race_match_x_party_match", "gender_match_x_party_match",
    "age_diff_x_party_dist", "educ_x_ideology_dist", "income_x_party_dist",
    "party_ideology_race_align", "demographic_similarity",
]
# 2024-style values after the prompt builder's defaults.
ABLATED_VALUES = {"party_match": -1, "party_alignment": 0}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def key(meta: dict) -> tuple[int, int, str]:
    return (int(meta["respondent_id"]), int(meta["year"]), str(meta["candidate"]))


def ablate_instruction(text: str) -> str:
    """Replace the four prompt lines that 2024 loses with the 2024 defaults."""
    text = re.sub(r"^- Education:.*$", "- Education: High school", text, flags=re.M)
    text = re.sub(
        r"^- Political:.*$", "- Political: Independent, Moderate", text, flags=re.M
    )
    text = re.sub(
        r"^- Party alignment:.*$",
        "- Party alignment: Independent voter",
        text,
        flags=re.M,
    )
    text = re.sub(
        r"^- Ideology distance:.*$",
        "- Ideology distance: 0.0 (on 7-point scale)",
        text,
        flags=re.M,
    )
    return text


def count_profiles(instructions: list[str]) -> tuple[int, str]:
    """Count distinct (education, political, alignment, distance) blocks."""
    profiles = set()
    example = ""
    for text in instructions:
        block = " | ".join(
            line for line in text.splitlines()
            if line.startswith(("- Education:", "- Political:",
                                "- Party alignment:", "- Ideology distance:"))
        )
        profiles.add(block)
    if profiles:
        example = sorted(profiles)[0]
    return len(profiles), example


def _metrics(df: pd.DataFrame) -> dict:
    good = df[df["pred"].notna()].copy()
    out = metrics.summarize(good) if len(good) else {}
    out["pred_std"] = float(good["pred"].std()) if len(good) else float("nan")
    out["pred_mean"] = float(good["pred"].mean()) if len(good) else float("nan")
    out["n_distinct_pred"] = int(good["pred"].nunique()) if len(good) else 0
    return out


def _load_rows(split_path: Path, year: int) -> list[dict]:
    rows = []
    for obj in load_jsonl(split_path):
        meta = obj["metadata"]
        if int(meta["year"]) != year:
            continue
        rows.append({"meta": meta, "instruction": obj["instruction"],
                     "true_score": float(meta["true_score"])})
    return rows


def _existing_preds(pred_path: Path, year: int) -> dict:
    out = {}
    for obj in load_jsonl(pred_path):
        if int(obj["year"]) != year:
            continue
        out[(int(obj["respondent_id"]), int(obj["year"]), str(obj["candidate"]))] = obj
    return out


def _frame(rows: list[dict], preds: list[dict], condition: str) -> pd.DataFrame:
    records = []
    for row, pred in zip(rows, preds):
        meta = row["meta"]
        records.append({
            "respondent_id": int(meta["respondent_id"]),
            "year": int(meta["year"]),
            "candidate": str(meta["candidate"]),
            "true_score": row["true_score"],
            "pred": pred,
            "condition": condition,
        })
    return pd.DataFrame(records)


def _tabular_predictions(rows: list[dict], ablate: bool) -> list[float]:
    clean = pd.read_csv(config.V3 / "anes_candidates_clean.csv", low_memory=False)
    clean = clean.set_index("orig_row_id", drop=False)
    train = load_jsonl(config.V3_SPLITS / "train_r.jsonl")
    train_ids = [int(obj["metadata"]["orig_row_id"]) for obj in train]
    train_rows = clean.loc[train_ids].reset_index(drop=True)
    X_train = _design(train_rows, CAT_COLS)
    y_train = np.array([float(obj["metadata"]["true_score"]) for obj in train])

    from sklearn.ensemble import HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor(
        max_iter=60, learning_rate=0.08, max_bins=64,
        early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=5, random_state=42,
    )
    model.fit(X_train, y_train)

    ids = [int(row["meta"]["orig_row_id"]) for row in rows]
    test_rows = clean.loc[ids].reset_index(drop=True)
    if ablate:
        test_rows = test_rows.copy()
        for col in POLITICAL_COLS:
            if col in test_rows.columns:
                test_rows[col] = np.nan
        for col, value in ABLATED_VALUES.items():
            if col in test_rows.columns:
                test_rows[col] = value
    X = _design(test_rows, CAT_COLS).reindex(columns=X_train.columns, fill_value=0)
    return list(np.clip(model.predict(X), 0, 100))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(config.V3_SPLITS / "test_r.jsonl"))
    p.add_argument("--preds", default=str(DEFAULT_PREDS),
                   help="committed seed-42 predictions for the original condition")
    p.add_argument("--year", type=int, default=2019)
    p.add_argument("--model", default="qwen3-8b")
    p.add_argument("--lora", default=str(DEFAULT_LORA))
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--skip-llm", action="store_true",
                   help="only validate prompts and tabular predictions")
    p.add_argument("--output-dir", default=str(config.RESULTS / "diagnostics"))
    args = p.parse_args()

    rows = _load_rows(Path(args.split), args.year)
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit(f"no rows for year {args.year} in {args.split}")

    preds = _existing_preds(Path(args.preds), args.year)
    missing = [key(row["meta"]) for row in rows if key(row["meta"]) not in preds]
    if missing:
        raise SystemExit(
            f"{len(missing)} rows have no committed prediction; first={missing[0]}"
        )

    original_instructions = [row["instruction"] for row in rows]
    ablated_instructions = [ablate_instruction(text) for text in original_instructions]
    n_orig, example_orig = count_profiles(original_instructions)
    n_abl, example_abl = count_profiles(ablated_instructions)
    if n_abl != 1:
        raise SystemExit(f"ablation produced {n_abl} profiles, expected 1")

    original_preds = [preds[key(row["meta"])]["pred"] for row in rows]
    original = _frame(rows, original_preds, "original")

    result = {
        "config": {
            "split": args.split,
            "preds": args.preds,
            "year": args.year,
            "n_rows": len(rows),
            "lora": args.lora,
            "model": args.model,
            "prompt_profiles_original": n_orig,
            "prompt_profiles_ablated": n_abl,
            "example_original": example_orig,
            "example_ablated": example_abl,
        },
        "llm": {"original": _metrics(original)},
        "tabular": {},
    }

    if not args.skip_llm:
        from experiments.eval.readout import RatingGenerator
        gen = RatingGenerator.load(args.model, args.lora, batch_size=args.batch_size)
        raw = gen.predict(
            ablated_instructions,
            task="candidate",
            max_new_tokens=args.max_new_tokens,
        )
        ablated_preds = [item["pred"] for item in raw]
        parse_rate = float(np.mean([v is not None for v in ablated_preds]))
        ablated = _frame(rows, ablated_preds, "ablated")
        result["llm"]["ablated"] = _metrics(ablated)
        result["llm"]["parse_rate"] = parse_rate
        result["llm"]["raw_examples"] = [item["text"] for item in raw[:5]]
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ablated.assign(raw=[item["text"] for item in raw]).to_json(
            out_dir / "covariate_dropout_probe_ablated.predictions.jsonl",
            orient="records", lines=True,
        )

    # Tabular probe: same recipe, same train split, original vs ablated features.
    tab_orig = _tabular_predictions(rows, ablate=False)
    tab_abl = _tabular_predictions(rows, ablate=True)
    result["tabular"]["original"] = _metrics(_frame(rows, tab_orig, "original"))
    result["tabular"]["ablated"] = _metrics(_frame(rows, tab_abl, "ablated"))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "covariate_dropout_probe.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    lines = [
        "# Covariate-dropout probe",
        "",
        f"Year {args.year}, `{Path(args.split).name}`, {len(rows):,} rows. "
        "The ablated prompt uses the 2024 defaults: Independent, Moderate, "
        "High school, Independent voter, ideology distance 0.0.",
        "",
        f"- distinct prompt profiles: original **{n_orig}**, ablated **{n_abl}**",
        "",
        "| Condition | Model | n | MAE | Pearson | choice | chance | pred std |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in ("llm", "tabular"):
        for condition in ("original", "ablated"):
            m = result.get(model_name, {}).get(condition)
            if not m:
                continue
            lines.append(
                f"| {condition} | {model_name.upper()} | {m.get('n', 0):,} | "
                f"{m.get('mae', float('nan')):.3f} | "
                f"{m.get('pearson', float('nan')):.3f} | "
                f"{m.get('choice', {}).get('accuracy', float('nan')):.3f} | "
                f"{m.get('choice', {}).get('chance', float('nan')):.3f} | "
                f"{m.get('pred_std', float('nan')):.2f} |"
            )
    (out_dir / "covariate_dropout_probe.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
