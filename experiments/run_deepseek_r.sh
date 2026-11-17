#!/bin/bash
# Second-checkpoint respondent-disjoint run: DeepSeek-R1-Distill-Llama-8B,
# trained on train_r, one seed, evaluated on test_r and test_t.
# Usage: bash experiments/run_deepseek_r.sh [GPU]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python}"
GPU="${1:-2}"
MODEL=deepseek-r1-distill-llama-8b
OUT=sft/sft_models/deepseek_r1_distill_8b_r_seed42

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/preds experiments/results/logs

echo "=== $(date '+%F %T') train ${MODEL} on train_r seed42 -> ${OUT}"
"$PY" -u -m experiments.train_lora \
  --model "$MODEL" \
  --train-file data/anes/v3/splits/train_r.jsonl \
  --output-dir "$OUT" \
  --seed 42 --epochs 3 --batch-size 4 --grad-accum 4 \
  --max-length 512 --gradient-checkpointing --lr 1e-4 \
  >experiments/results/logs/train_deepseek_r.log 2>&1

for split in test_r test_t; do
  echo "=== $(date '+%F %T') eval ${split}"
  "$PY" -u -m experiments.eval.evaluate \
    --split "data/anes/v3/splits/${split}.jsonl" \
    --model "$MODEL" --lora "$OUT" \
    --readout generate --max-new-tokens 16 --batch-size 8 \
    --output "experiments/results/preds/deepseek_r_seed42_${split}_generate.jsonl" \
    >"experiments/results/logs/eval_deepseek_r_${split}.log" 2>&1
done

echo "=== $(date '+%F %T') DONE"
