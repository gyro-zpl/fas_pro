#!/bin/bash
#
# ANES群体态度预测 - 快速测试版（1个epoch）
#

echo "================================"
echo "ANES群体态度预测 - 快速测试"
echo "================================"

cd /home/cyyuan/FAS_PRO/sft

TRAIN_FILE="/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_train_enhanced.jsonl"
EVAL_FILE="/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_enhanced.jsonl"
OUTPUT_DIR="/home/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora_quick"

echo ""
echo "快速测试模式:"
echo "  仅训练 1 epoch"
echo "  Batch size: 2 × 2"
echo "  每50步评估一次"
echo ""

python finetune_anes_groups.py \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --save_steps 50 \
    --eval_steps 50 \
    --logging_steps 5

echo ""
echo "✅ 快速测试完成！"
echo ""
echo "如果一切正常，运行完整训练:"
echo "  bash scripts/run_finetune_groups.sh"

