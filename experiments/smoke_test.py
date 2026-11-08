"""Minimal end-to-end check: load model, prompt, generate, parse.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.smoke_test
"""
from __future__ import annotations

import json
import time

from experiments import config
from experiments.eval.readout import RatingGenerator, inspect_rating_tokens


def main() -> None:
    split = config.V3_SPLITS / "train_t.jsonl"
    if not split.exists():
        raise SystemExit(f"Missing split: {split}. Run build_splits first.")

    with open(split) as f:
        samples = [json.loads(next(f)) for _ in range(3)]

    print(f"model: {config.DEFAULT_MODEL} -> {config.model_info()['path']}")
    t0 = time.time()
    gen = RatingGenerator.load(batch_size=3)
    print(f"loaded in {time.time() - t0:.1f}s")
    print("tokenization:", json.dumps(inspect_rating_tokens(gen.tokenizer), indent=2)[:400])

    t0 = time.time()
    out = gen.predict([s["instruction"] for s in samples])
    print(f"generated in {time.time() - t0:.1f}s")
    for sample, res in zip(samples, out):
        true = sample["metadata"]["true_score"]
        print("-" * 60)
        print("prompt tail:", sample["instruction"][-120:].replace("\n", " "))
        print("raw:", repr(res["text"][:120]))
        print(f"parsed: {res['pred']} | true: {true}")


if __name__ == "__main__":
    main()
