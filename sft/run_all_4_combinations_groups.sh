#!/bin/bash

# 测试4种组合的脚本 - Groups版本
# 1. Base Model + Zero-shot
# 2. Base Model + Few-shot
# 3. Fine-tuned + Zero-shot
# 4. Fine-tuned + Few-shot

set -e  # 遇到错误立即退出

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH="/home/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora/final_model"
TEST_FILE="/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_balanced_v3.jsonl"
OUTPUT_DIR="/home/cyyuan/FAS_PRO/data/anes/extend_features/results"

echo "=========================================="
echo "运行4种组合测试 - Groups"
echo "=========================================="

# 1. Base Model + Zero-shot
echo ""
echo "🔵 [1/4] Base Model + Zero-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes_groups.py \
    --base_model "${BASE_MODEL}" \
    --model_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --base_model_only

# 2. Base Model + Few-shot
echo ""
echo "🟢 [2/4] Base Model + Few-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes_groups.py \
    --base_model "${BASE_MODEL}" \
    --model_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --base_model_only \
    --use_few_shot

# 3. Fine-tuned + Zero-shot
echo ""
echo "🟡 [3/4] Fine-tuned + Zero-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes_groups.py \
    --base_model "${BASE_MODEL}" \
    --model_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}"

# 4. Fine-tuned + Few-shot  
echo ""
echo "🔴 [4/4] Fine-tuned + Few-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes_groups.py \
    --base_model "${BASE_MODEL}" \
    --model_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    --use_few_shot

echo ""
echo "=========================================="
echo "✅ 所有4种组合测试完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  1. ${OUTPUT_DIR}/group_predictions_base_zero_shot.jsonl"
echo "  2. ${OUTPUT_DIR}/group_predictions_base_few_shot.jsonl"
echo "  3. ${OUTPUT_DIR}/group_predictions_finetuned_zero_shot.jsonl"
echo "  4. ${OUTPUT_DIR}/group_predictions_finetuned_few_shot.jsonl"
echo ""
echo "对应的summary文件:"
echo "  1. ${OUTPUT_DIR}/group_evaluation_summary_base_zero_shot.json"
echo "  2. ${OUTPUT_DIR}/group_evaluation_summary_base_few_shot.json"
echo "  3. ${OUTPUT_DIR}/group_evaluation_summary_finetuned_zero_shot.json"
echo "  4. ${OUTPUT_DIR}/group_evaluation_summary_finetuned_few_shot.json"

