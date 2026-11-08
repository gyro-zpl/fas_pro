"""Evaluate a model on a clean split.

Example:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.eval.evaluate \
        --split data/anes/v3/splits/test_t.jsonl \
        --model qwen3-8b --prompt-mode zero --limit 100 \
        --output experiments/results/preds/qwen3_test_t_zero.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from experiments import config
from experiments.eval import metrics
from experiments.eval.readout import RatingGenerator


def _load(path: str | Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f]


def _breakdown(df: pd.DataFrame, field: str) -> dict:
    if field not in df.columns:
        return {}
    out = {}
    for value, g in df.groupby(field):
        out[str(value)] = {
            "n": int(len(g)),
            "mae": metrics.mae(g["true_score"], g["pred"]),
            "rmse": metrics.rmse(g["true_score"], g["pred"]),
            "relaxed_acc": metrics.relaxed_accuracy(g["true_score"], g["pred"]),
        }
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", required=True)
    p.add_argument("--model", default=config.DEFAULT_MODEL)
    p.add_argument("--lora", default=None)
    p.add_argument("--task", default="candidate", choices=["candidate", "group"])
    p.add_argument("--prompt-mode", default="zero", choices=["zero", "few"])
    p.add_argument("--readout", default="anchor", choices=["anchor", "generate"],
                   help="anchor = 11-anchor likelihood (fast, calibrated); "
                        "generate = free generation + parse")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--anchor-batch", type=int, default=3,
                   help="samples per forward in anchor readout")
    p.add_argument("--output", required=True)
    args = p.parse_args()

    samples = _load(args.split)
    if args.limit:
        samples = samples[: args.limit]

    gen = RatingGenerator.load(args.model, args.lora, batch_size=args.batch_size)
    if args.readout == "anchor":
        from experiments.eval.anchor_readout import anchor_predict
        raw = anchor_predict(
            gen, [s["instruction"] for s in samples],
            task=args.task, few_shot=(args.prompt_mode == "few"),
            samples_per_batch=args.anchor_batch,
        )
        results = [{"text": f"anchor_expectation={r['expectation']:.2f}",
                    "pred": int(round(r["expectation"])),
                    "pred_float": float(r["expectation"])} for r in raw]
    else:
        results = gen.predict(
            [s["instruction"] for s in samples],
            few_shot=(args.prompt_mode == "few"),
            task=args.task,
        )

    rows = []
    for sample, result in zip(samples, results):
        meta = sample.get("metadata", {})
        rows.append({
            "respondent_id": meta.get("respondent_id"),
            "year": meta.get("year"),
            "candidate": meta.get("candidate"),
            "group": meta.get("group"),
            "strategy": meta.get("strategy"),
            "true_score": float(meta.get("true_score", sample.get("output"))),
            "pred": result["pred"],
            "pred_float": result.get("pred_float", result["pred"]),
            "raw": result["text"],
            "instruction": sample["instruction"],
        })
    df = pd.DataFrame(rows)

    parse_rate = float(df["pred"].notna().mean())
    good = df[df["pred"].notna()].copy()
    summary = {
        "config": {
            "split": str(args.split),
            "model": args.model,
            "lora": args.lora,
            "task": args.task,
            "prompt_mode": args.prompt_mode,
            "n_requested": len(samples),
            "n_parsed": int(len(good)),
            "parse_rate": parse_rate,
        },
        "overall": metrics.summarize(good) if len(good) else {},
        "by_candidate": _breakdown(good, "candidate"),
        "by_year": _breakdown(good, "year"),
        "by_strategy": _breakdown(good, "strategy"),
        "by_group": _breakdown(good, "group"),
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out.with_suffix(".predictions.jsonl"), orient="records", lines=True)
    with open(out.with_suffix(".summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["config"], indent=2))
    print(json.dumps(summary["overall"], indent=2))
    print(f"\nSaved {out.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
