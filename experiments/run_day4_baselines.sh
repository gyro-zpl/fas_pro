#!/bin/bash
# D4: base-model evaluations on the clean splits (Qwen3-8B, no LoRA yet).
# Usage: bash experiments/run_day4_baselines.sh
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
export CUDA_VISIBLE_DEVICES=3
cd "$ROOT"

mkdir -p experiments/results/preds experiments/results/logs

for split in test_t test_tr test_r; do
  for mode in zero few; do
    out="experiments/results/preds/qwen3_${split}_${mode}.jsonl"
    echo "=== $(date '+%F %T') ${split} ${mode} -> ${out}"
    "$PY" -m experiments.eval.evaluate \
      --split "data/anes/v3/splits/${split}.jsonl" \
      --model qwen3-8b \
      --prompt-mode "$mode" \
      --batch-size 16 \
      --output "$out"
  done
done

echo "=== $(date '+%F %T') DONE"
