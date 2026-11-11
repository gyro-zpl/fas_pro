#!/bin/bash
# DeepSeek second-family run on a free GPU, no gradient checkpointing.
# Usage: bash experiments/run_deepseek_fast.sh [GPU]
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
GPU="${1:-3}"
MODEL=deepseek-r1-distill-llama-8b
OUT=sft/sft_models/deepseek_r1_distill_8b_t_seed42

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/preds experiments/results/logs

echo "=== $(date '+%F %T') train ${MODEL} seed42 -> ${OUT}"
"$PY" -u -m experiments.train_lora \
  --model "$MODEL" \
  --train-file data/anes/v3/splits/train_t.jsonl \
  --output-dir "$OUT" \
  --seed 42 --epochs 3 --batch-size 8 --grad-accum 2 \
  --max-length 512 --lr 1e-4

for split in test_r test_t; do
  echo "=== $(date '+%F %T') eval ${split}"
  "$PY" -u -m experiments.eval.evaluate \
    --split "data/anes/v3/splits/${split}.jsonl" \
    --model "$MODEL" --lora "$OUT" \
    --readout generate --max-new-tokens 16 --batch-size 8 \
    --output "experiments/results/preds/deepseek_${split}_generate.jsonl"
done

echo "=== $(date '+%F %T') DONE"
