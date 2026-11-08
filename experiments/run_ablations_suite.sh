#!/bin/bash
# Run the three ablation trainings + evaluations sequentially on one GPU.
# Usage: bash experiments/run_ablations_suite.sh GPU
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
GPU="${1:?GPU}"
cd "$ROOT"

bash experiments/run_one_train_eval.sh "$GPU" a1_random \
  data/anes/v3/ablations/train_a1_random.jsonl \
  data/anes/v3/splits/test_r.jsonl \
  sft/sft_models/qwen3_8b_a1_random

bash experiments/run_one_train_eval.sh "$GPU" a2_minimal \
  data/anes/v3/ablations/train_a2_minimal.jsonl \
  data/anes/v3/ablations/test_r_a2_minimal.jsonl \
  sft/sft_models/qwen3_8b_a2_minimal

bash experiments/run_one_train_eval.sh "$GPU" a3_name_only \
  data/anes/v3/ablations/train_a3_name_only.jsonl \
  data/anes/v3/ablations/test_r_a3_name_only.jsonl \
  sft/sft_models/qwen3_8b_a3_name_only

echo "=== $(date '+%F %T') ABLATIONS DONE"
