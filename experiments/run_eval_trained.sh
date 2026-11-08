#!/bin/bash
# Evaluate the 3 temporal-protocol LoRA seeds on test_t / test_tr / test_r.
# Usage: bash experiments/run_eval_trained.sh
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
export CUDA_VISIBLE_DEVICES=3
cd "$ROOT"

mkdir -p experiments/results/preds experiments/results/logs

for seed in 42 43 44; do
  lora="sft/sft_models/qwen3_8b_t_seed${seed}"
  [ -d "$lora" ] || { echo "missing $lora"; continue; }
  for split in test_t test_tr test_r; do
    out="experiments/results/preds/qwen3_t_seed${seed}_${split}_anchor.jsonl"
    echo "=== $(date '+%F %T') seed ${seed} ${split}"
    "$PY" -u -m experiments.eval.evaluate \
      --split "data/anes/v3/splits/${split}.jsonl" \
      --model qwen3-8b \
      --lora "$lora" \
      --prompt-mode zero \
      --readout anchor \
      --anchor-batch 3 \
      --output "$out"
  done
done

echo "=== $(date '+%F %T') DONE"
