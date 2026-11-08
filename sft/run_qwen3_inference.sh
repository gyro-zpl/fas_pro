#!/bin/bash

# Qwen3模型推理脚本
# 测试zero-shot和few-shot两种模式

set -e

BASE_DIR="/home/cyyuan/FAS_PRO/sft"
OUTPUT_DIR="${BASE_DIR}/sft_models/qwen3_results"
TEST_FILE="/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl"

# 创建输出目录
mkdir -p "${OUTPUT_DIR}"

echo "=========================================="
echo "Qwen3模型ANES推理"
echo "=========================================="
echo ""

# 测试1: Zero-shot
echo "🔵 测试1: Qwen3 + Zero-shot"
echo "=========================================="
python ${BASE_DIR}/inference_anes_qwen3.py \
    --test_file "${TEST_FILE}" \
    --output_file "${OUTPUT_DIR}/qwen3_predictions_zero_shot.jsonl" \
    --summary_file "${OUTPUT_DIR}/qwen3_predictions_zero_shot_summary.json" \
    --temperature 0.3

echo ""
echo "✅ Zero-shot测试完成！"
echo ""

# 测试2: Few-shot
echo "🟢 测试2: Qwen3 + Few-shot"
echo "=========================================="
python ${BASE_DIR}/inference_anes_qwen3.py \
    --test_file "${TEST_FILE}" \
    --output_file "${OUTPUT_DIR}/qwen3_predictions_few_shot.jsonl" \
    --summary_file "${OUTPUT_DIR}/qwen3_predictions_few_shot_summary.json" \
    --temperature 0.3 \
    --use_few_shot

echo ""
echo "✅ Few-shot测试完成！"
echo ""

echo "=========================================="
echo "所有测试完成！"
echo "=========================================="
echo ""
echo "结果文件位置:"
echo "  Zero-shot: ${OUTPUT_DIR}/qwen3_predictions_zero_shot_summary.json"
echo "  Few-shot: ${OUTPUT_DIR}/qwen3_predictions_few_shot_summary.json"
echo ""

