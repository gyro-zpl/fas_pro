"""Compare readouts on a fine-tuned checkpoint.

Conditions: generate (no thinking), anchor likelihood, generate (thinking).
Uses 100 complete respondent groups (both candidates) from test_t.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.diagnostics.readout_compare \
        --lora sft/sft_models/qwen3_8b_t_seed42
"""
from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments import config
from experiments.eval import metrics
from experiments.eval.anchor_readout import anchor_predict
from experiments.eval.readout import RatingGenerator


def load_groups(n_groups: int = 100) -> list[dict]:
    rows = [json.loads(line) for line in open(config.V3_SPLITS / "test_t.jsonl")]
    by_resp: dict[int, list[dict]] = {}
    for r in rows:
        by_resp.setdefault(r["metadata"]["respondent_id"], []).append(r)
    complete = [v for v in by_resp.values()
                if {x["metadata"]["candidate"] for x in v} == {"biden", "trump"}]
    sample = []
    for group in complete[:n_groups]:
        sample.extend(group)
    return sample


def report(name: str, sample: list[dict], preds: list[float]) -> None:
    df = pd.DataFrame({
        "respondent_id": [s["metadata"]["respondent_id"] for s in sample],
        "year": [s["metadata"]["year"] for s in sample],
        "true_score": [s["metadata"]["true_score"] for s in sample],
        "pred": preds,
    })
    print(f"\n=== {name}")
    print(f"MAE {metrics.mae(df.true_score, df.pred):.2f} | "
          f"pred mean {df.pred.mean():.1f} std {df.pred.std():.1f} | "
          f"true mean {df.true_score.mean():.1f}")
    print("choice:", metrics.choice_accuracy(df))
    print(df.head(6).to_string(index=False))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--lora", required=True)
    p.add_argument("--n-groups", type=int, default=100)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--skip-thinking", action="store_true")
    args = p.parse_args()

    sample = load_groups(args.n_groups)
    texts = [s["instruction"] for s in sample]
    print(f"samples: {len(sample)}")

    gen = RatingGenerator.load(args.model, args.lora, batch_size=16)

    out = gen.predict(texts, enable_thinking=False)
    report("generate / no-think", sample, [r["pred"] for r in out])

    anc = anchor_predict(gen, texts, samples_per_batch=3)
    report("anchor", sample, [r["expectation"] for r in anc])

    if not args.skip_thinking:
        out = gen.predict(texts, enable_thinking=True, max_new_tokens=256)
        report("generate / think", sample, [r["pred"] for r in out])


if __name__ == "__main__":
    main()
