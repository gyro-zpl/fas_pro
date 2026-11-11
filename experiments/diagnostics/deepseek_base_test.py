"""One-sample check of the DeepSeek-R1-Distill base model output format."""
from __future__ import annotations

import json

from experiments.eval.readout import RatingGenerator


def main() -> None:
    gen = RatingGenerator.load("deepseek-r1-distill-llama-8b", batch_size=1)
    row = json.loads(open("data/anes/v3/splits/train_t.jsonl").readline())
    out = gen.predict([row["instruction"]], max_new_tokens=256)[0]
    print("TRUE", row["metadata"]["true_score"])
    print("RAW_TAIL", repr(out["text"][-600:]))
    print("PARSED", out["pred"])


if __name__ == "__main__":
    main()
