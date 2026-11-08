#!/bin/bash

# 优化的训练配置（解决loss不下降问题）

python finetune_anes_groups.py \
  --learning_rate 3e-5 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --num_train_epochs 3 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 50 \
  --warmup_steps 500 \
  --max_length 512

echo ""
echo "关键调整:"
echo "  - 学习率: 1e-4 → 3e-5 (降低70%，提高稳定性)"
echo "  - 有效batch: 16 → 32 (提高梯度估计质量)"
echo "  - Warmup: 无 → 500步 (平滑启动)"
