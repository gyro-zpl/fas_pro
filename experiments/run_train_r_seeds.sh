#!/bin/bash
# Train LoRA for the respondent-disjoint protocol (train_r), 3 seeds.
# Usage: bash experiments/run_train_r_seeds.sh
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
export CUDA_VISIBLE_DEVICES=3
cd "$ROOT"

mkdir -p experiments/results/logs

for seed in 42 43 44; do
  out="sft/sft_models/qwen3_8b_r_seed${seed}"
  echo "=== $(date '+%F %T') train_r seed ${seed} -> ${out}"
  "$PY" -u -m experiments.train_lora \
    --train-file data/anes/v3/splits/train_r.jsonl \
    --output-dir "$out" \
    --seed "$seed" \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-length 512 \
    --gradient-checkpointing \
    --lr 1e-4
done

echo "=== $(date '+%F %T') DONE"
