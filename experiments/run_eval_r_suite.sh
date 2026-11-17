#!/bin/bash
# Evaluate respondent-protocol LoRA models on test_r / test_t.
# Usage: bash experiments/run_eval_r_suite.sh GPU seed:split:readout ...
# Example: bash experiments/run_eval_r_suite.sh 0 42:test_r:generate 42:test_t:generate
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python}"
GPU="${1:?usage: $0 GPU seed:split:readout ...}"
shift

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/preds experiments/results/logs

for spec in "$@"; do
  IFS=: read -r seed split readout <<< "$spec"
  out="experiments/results/preds/qwen3_r_seed${seed}_${split}_${readout}.jsonl"
  log="experiments/results/logs/eval_r_seed${seed}_${split}_${readout}.log"
  echo "=== $(date '+%F %T') GPU${GPU} seed${seed} ${split} ${readout}"
  "$PY" -u -m experiments.eval.evaluate \
    --split "data/anes/v3/splits/${split}.jsonl" \
    --model qwen3-8b \
    --lora "sft/sft_models/qwen3_8b_r_seed${seed}" \
    --prompt-mode zero \
    --readout "$readout" \
    --anchor-batch 3 \
    --output "$out" >"$log" 2>&1
done

echo "=== $(date '+%F %T') GPU${GPU} DONE"
