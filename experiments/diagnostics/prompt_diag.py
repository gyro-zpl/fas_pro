"""Small-sample prompt diagnostics for the rating task.

Compares four prompting/decoding conditions on the same 60 test_t rows.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.diagnostics.prompt_diag
"""
from __future__ import annotations

import json

import pandas as pd

from experiments import config
from experiments.eval import metrics
from experiments.eval.readout import RatingGenerator

CALIB = (
    "Scores are not a general favorability rating. 0 means the voter strongly "
    "opposes the candidate, 50 is neutral, and 100 means strong support. "
    "Use the full 0-100 range and do not default to high scores."
)


def run(gen, samples, texts, name, **kwargs):
    out = gen.predict(texts, **kwargs)
    df = pd.DataFrame({
        "true_score": [s["metadata"]["true_score"] for s in samples],
        "pred": [r["pred"] for r in out],
        "raw": [r["text"] for r in out],
        "respondent_id": [s["metadata"]["respondent_id"] for s in samples],
        "year": [s["metadata"]["year"] for s in samples],
    })
    good = df[df["pred"].notna()]
    print(f"\n=== {name}")
    print(f"parse {len(good)}/{len(df)} | MAE {metrics.mae(good.true_score, good.pred):.2f} "
          f"| pred mean {good.pred.mean():.1f} std {good.pred.std():.1f} "
          f"| true mean {good.true_score.mean():.1f}")
    print("choice:", metrics.choice_accuracy(good))
    print("pred top:", good.pred.value_counts().head(5).to_dict())
    print("raw head:", good.raw.head(5).tolist())
    return df


def main() -> None:
    with open(config.V3_SPLITS / "test_t.jsonl") as f:
        samples = [json.loads(next(f)) for _ in range(60)]
    texts = [s["instruction"] for s in samples]

    gen = RatingGenerator.load(batch_size=16)
    run(gen, samples, texts, "zero / no-think")
    run(gen, samples, texts, "zero / no-think / calibration", extra_system=CALIB)
    run(gen, samples, texts, "few / no-think", few_shot=True)
    run(gen, samples, texts, "zero / think", enable_thinking=True, max_new_tokens=256)


if __name__ == "__main__":
    main()
