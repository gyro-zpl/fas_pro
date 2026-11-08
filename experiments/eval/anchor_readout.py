"""Anchor-likelihood readout for calibrated rating prediction.

Instead of free generation (which Qwen3 skews high when thinking is disabled),
score the likelihood of 11 anchor strings "0", "10", ..., "100" given the prompt
and take the softmax over anchors. This is a verbalizer-style readout: one
forward pass over `len(anchors)` copies of each prompt, no generation.

Usage:
    CUDA_VISIBLE_DEVICES=3 python -m experiments.eval.anchor_readout --limit 60
"""
from __future__ import annotations

import argparse
import json

import torch
import torch.nn.functional as F

from experiments import config
from experiments.eval.readout import (
    RatingGenerator, apply_chat, build_messages,
)
from experiments.eval import metrics

ANCHORS = list(range(0, 101, 10))


@torch.no_grad()
def anchor_predict(gen: RatingGenerator, instructions: list[str],
                   anchors: list[int] = ANCHORS, task: str = "candidate",
                   few_shot: bool = False, samples_per_batch: int = 2) -> list[dict]:
    tok = gen.tokenizer
    model = gen.model
    results = []

    for start in range(0, len(instructions), samples_per_batch):
        batch = instructions[start:start + samples_per_batch]
        seqs, meta = [], []
        for text in batch:
            prompt = apply_chat(tok, build_messages(text, few_shot, task),
                                gen.family, enable_thinking=False)
            prompt_ids = tok(prompt, add_special_tokens=False).input_ids
            for a in anchors:
                a_ids = tok(str(a), add_special_tokens=False).input_ids
                seqs.append((prompt_ids, a_ids))
                meta.append((len(prompt_ids), len(a_ids)))

        # Left-pad to the max length in this batch.
        max_len = max(len(p) + len(a) for p, a in seqs)
        input_ids = torch.full((len(seqs), max_len), tok.pad_token_id, dtype=torch.long)
        attn = torch.zeros((len(seqs), max_len), dtype=torch.long)
        for i, (p, a) in enumerate(seqs):
            ids = p + a
            input_ids[i, max_len - len(ids):] = torch.tensor(ids)
            attn[i, max_len - len(ids):] = 1
        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)

        logits = model(input_ids=input_ids, attention_mask=attn).logits
        logp = F.log_softmax(logits.float(), dim=-1)

        # Gather the log-probability of each anchor token.
        per_seq = []
        for i, (p, a) in enumerate(seqs):
            offset = max_len - (len(p) + len(a))
            total = 0.0
            for j, token_id in enumerate(a):
                pos = offset + len(p) + j - 1  # next-token position
                total += float(logp[i, pos, token_id])
            per_seq.append(total)

        n = len(anchors)
        for k in range(len(batch)):
            vals = torch.tensor(per_seq[k * n:(k + 1) * n])
            probs = F.softmax(vals, dim=0)
            exp = float(sum(p * a for p, a in zip(probs.tolist(), anchors)))
            results.append({
                "expectation": exp,
                "argmax": anchors[int(torch.argmax(probs))],
                "probs": {a: float(p) for a, p in zip(anchors, probs.tolist())},
            })
    return results


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--split", default=str(config.V3_SPLITS / "test_t.jsonl"))
    p.add_argument("--limit", type=int, default=60)
    p.add_argument("--per-candidate", type=int, default=30)
    p.add_argument("--few-shot", action="store_true")
    args = p.parse_args()

    rows = [json.loads(line) for line in open(args.split)]
    # The split is sorted by candidate; take a balanced sample for diagnostics.
    biden = [r for r in rows if r["metadata"].get("candidate") == "biden"][: args.per_candidate]
    trump = [r for r in rows if r["metadata"].get("candidate") == "trump"][: args.per_candidate]
    sample = biden + trump

    gen = RatingGenerator.load(batch_size=1)
    out = anchor_predict(gen, [s["instruction"] for s in sample],
                         few_shot=args.few_shot)

    import pandas as pd
    df = pd.DataFrame({
        "respondent_id": [s["metadata"]["respondent_id"] for s in sample],
        "year": [s["metadata"]["year"] for s in sample],
        "candidate": [s["metadata"]["candidate"] for s in sample],
        "true_score": [s["metadata"]["true_score"] for s in sample],
        "pred": [r["expectation"] for r in out],
        "argmax": [r["argmax"] for r in out],
    })
    print(f"MAE {metrics.mae(df.true_score, df.pred):.2f} | "
          f"pred mean {df.pred.mean():.1f} std {df.pred.std():.1f} | "
          f"true mean {df.true_score.mean():.1f}")
    print("choice:", metrics.choice_accuracy(df))
    print(df.head(12).to_string(index=False))
    df.to_json(config.RESULTS / "anchor_diag.jsonl", orient="records", lines=True)


if __name__ == "__main__":
    main()
