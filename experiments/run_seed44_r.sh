#!/bin/bash
# Retry seed44 for the respondent protocol and evaluate it.
# The previous run was killed at 940/957 steps; move the partial dir aside.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python}"
GPU="${1:-1}"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/logs experiments/results/preds

if [ -d sft/sft_models/qwen3_8b_r_seed44 ]; then
  mv sft/sft_models/qwen3_8b_r_seed44 \
     "sft/sft_models/qwen3_8b_r_seed44_partial_$(date +%Y%m%d_%H%M%S)"
fi

echo "=== $(date '+%F %T') train seed44 on GPU${GPU}"
"$PY" -u -m experiments.train_lora \
  --train-file data/anes/v3/splits/train_r.jsonl \
  --output-dir sft/sft_models/qwen3_8b_r_seed44 \
  --seed 44 --epochs 3 --batch-size 4 --grad-accum 4 \
  --max-length 512 --gradient-checkpointing --lr 1e-4 \
  >experiments/results/logs/train_r_seed44_retry.log 2>&1

for split in test_r test_t; do
  echo "=== $(date '+%F %T') eval seed44 ${split}"
  "$PY" -u -m experiments.eval.evaluate \
    --split "data/anes/v3/splits/${split}.jsonl" \
    --model qwen3-8b \
    --lora sft/sft_models/qwen3_8b_r_seed44 \
    --prompt-mode zero --readout generate --anchor-batch 3 \
    --output "experiments/results/preds/qwen3_r_seed44_${split}_generate.jsonl" \
    >"experiments/results/logs/eval_r_seed44_${split}_generate.log" 2>&1
done

echo "=== $(date '+%F %T') DONE"
