#!/bin/bash
# Train one model then evaluate it on one split.
# Usage: bash experiments/run_one_train_eval.sh GPU name train_file eval_file output_dir
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
GPU="${1:?GPU}"; NAME="${2:?name}"; TRAIN="${3:?train}"; EVAL="${4:?eval}"; OUT="${5:?output}"

cd "$ROOT"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p experiments/results/logs experiments/results/preds

echo "=== $(date '+%F %T') GPU${GPU} train ${NAME} -> ${OUT}"
"$PY" -u -m experiments.train_lora \
  --train-file "$TRAIN" --output-dir "$OUT" --seed 42 \
  --epochs 3 --batch-size 8 --grad-accum 2 --max-length 512 --lr 1e-4

echo "=== $(date '+%F %T') GPU${GPU} eval ${NAME} on ${EVAL}"
"$PY" -u -m experiments.eval.evaluate \
  --split "$EVAL" --model qwen3-8b --lora "$OUT" \
  --prompt-mode zero --readout generate \
  --output "experiments/results/preds/${NAME}.jsonl"

echo "=== $(date '+%F %T') GPU${GPU} ${NAME} DONE"
