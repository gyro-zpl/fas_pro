#!/bin/bash
# Resume seed42 from checkpoint-600, then train seeds 43 and 44.
# Detached launch:
#   setsid nohup bash experiments/resume_train_seeds.sh \
#     > experiments/results/logs/train_resume.log 2>&1 < /dev/null &
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$ROOT"

mkdir -p experiments/results/logs

# --- seed 42: resume if the final adapter is absent ---
if [ ! -f "sft/sft_models/qwen3_8b_t_seed42/adapter_model.safetensors" ]; then
  echo "=== $(date '+%F %T') resume seed 42 from checkpoint-600"
  "$PY" -u -m experiments.train_lora \
    --train-file data/anes/v3/splits/train_t.jsonl \
    --output-dir sft/sft_models/qwen3_8b_t_seed42 \
    --seed 42 --epochs 3 --batch-size 8 --grad-accum 2 \
    --max-length 512 --lr 1e-4 \
    --resume-from-checkpoint sft/sft_models/qwen3_8b_t_seed42/checkpoint-600
fi

# --- seeds 43 and 44 from scratch ---
for seed in 43 44; do
  out="sft/sft_models/qwen3_8b_t_seed${seed}"
  echo "=== $(date '+%F %T') train seed ${seed} -> ${out}"
  "$PY" -u -m experiments.train_lora \
    --train-file data/anes/v3/splits/train_t.jsonl \
    --output-dir "$out" \
    --seed "$seed" --epochs 3 --batch-size 8 --grad-accum 2 \
    --max-length 512 --lr 1e-4
done

echo "=== $(date '+%F %T') DONE"
