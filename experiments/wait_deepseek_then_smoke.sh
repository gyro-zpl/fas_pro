#!/bin/bash
# Wait for the range downloader, then run a 5-sample smoke test on GPU 0.
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
cd "$ROOT"

while pgrep -f "experiments.download_deepseek_ranges" >/dev/null; do
  sleep 30
done

D=/home/jpzhao_team/models/deepseek-r1-distill-llama-8b
for f in model-00001-of-000002.safetensors model-00002-of-000002.safetensors; do
  if [ ! -f "$D/$f" ]; then echo "missing $f"; exit 1; fi
done
echo "=== $(date '+%F %T') download complete; smoke test"
CUDA_VISIBLE_DEVICES=0 "$PY" -u - <<'EOF'
import json
from experiments.eval.readout import RatingGenerator

gen = RatingGenerator.load("deepseek-r1-distill-llama-8b", batch_size=2)
rows = [json.loads(line) for line in list(open("data/anes/v3/splits/train_t.jsonl"))[:5]]
out = gen.predict([r["instruction"] for r in rows], max_new_tokens=16)
for r, o in zip(rows, out):
    print("TRUE", r["metadata"]["true_score"], "| RAW", repr(o["text"][:200]), "| PARSED", o["pred"])
EOF
echo "=== $(date '+%F %T') smoke done"
