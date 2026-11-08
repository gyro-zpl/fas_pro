#!/bin/bash
# Run a list of fine-tuned evaluations on one GPU.
# Usage: bash experiments/run_eval_suite.sh GPU seed:split:readout ...
# Example: bash experiments/run_eval_suite.sh 3 42:test_t:generate 42:test_r:generate
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
GPU="${1:?usage: $0 GPU seed:split:readout ...}"
shift

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/preds experiments/results/logs

for spec in "$@"; do
  IFS=: read -r seed split readout <<< "$spec"
  out="experiments/results/preds/qwen3_t_seed${seed}_${split}_${readout}.jsonl"
  echo "=== $(date '+%F %T') GPU${GPU} seed${seed} ${split} ${readout}"
  "$PY" -u -m experiments.eval.evaluate \
    --split "data/anes/v3/splits/${split}.jsonl" \
    --model qwen3-8b \
    --lora "sft/sft_models/qwen3_8b_t_seed${seed}" \
    --prompt-mode zero \
    --readout "$readout" \
    --anchor-batch 3 \
    --output "$out"
done

echo "=== $(date '+%F %T') GPU${GPU} DONE"
