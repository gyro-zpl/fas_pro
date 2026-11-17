#!/bin/bash
# Train the respondent-disjoint protocol (train_r) with 3 seeds in parallel.
# Usage: bash experiments/run_train_r_parallel.sh
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python}"
cd "$ROOT"
mkdir -p experiments/results/logs sft/sft_models

run_one() {
  local gpu="$1" seed="$2"
  local out="sft/sft_models/qwen3_8b_r_seed${seed}"
  local log="experiments/results/logs/train_r_seed${seed}.log"
  echo "=== $(date '+%F %T') GPU${gpu} train_r seed ${seed} -> ${out} (log ${log})"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" -u -m experiments.train_lora \
    --train-file data/anes/v3/splits/train_r.jsonl \
    --output-dir "$out" \
    --seed "$seed" \
    --epochs 3 \
    --batch-size 4 \
    --grad-accum 4 \
    --max-length 512 \
    --gradient-checkpointing \
    --lr 1e-4 >"$log" 2>&1
  echo "=== $(date '+%F %T') DONE seed ${seed}"
}

run_one 0 42 &
p42=$!
run_one 1 43 &
p43=$!
run_one 3 44 &
p44=$!

wait "$p42" "$p43" "$p44"
echo "=== $(date '+%F %T') ALL DONE"
