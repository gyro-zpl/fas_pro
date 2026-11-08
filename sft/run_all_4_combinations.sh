#!/bin/bash

# 测试4种组合的脚本
# 1. Base Model + Zero-shot
# 2. Base Model + Few-shot
# 3. Fine-tuned + Zero-shot
# 4. Fine-tuned + Few-shot

set -e  # 遇到错误立即退出

BASE_MODEL="meta-llama/Llama-3.1-8B-Instruct"
LORA_PATH="/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/final_model"
TEST_FILE="/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl"
OUTPUT_DIR="/home/cyyuan/FAS_PRO/sft/sft_models/jsons"
OUTPUT_FILE="${OUTPUT_DIR}/enhanced_test_dataset_predictions.jsonl"
TEMPERATURE=0.3

echo "=========================================="
echo "运行2种组合测试"
echo "=========================================="


# 3. Fine-tuned + Zero-shot
echo ""
echo "🟡 [1/2] Fine-tuned + Zero-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes.py \
    --base_model "${BASE_MODEL}" \
    --lora_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --temperature ${TEMPERATURE}

# 4. Fine-tuned + Few-shot  
echo ""
echo "🔴 [2/2] Fine-tuned + Few-shot"
python /home/cyyuan/FAS_PRO/sft/inference_anes.py \
    --base_model "${BASE_MODEL}" \
    --lora_path "${LORA_PATH}" \
    --test_file "${TEST_FILE}" \
    --output_file "${OUTPUT_FILE}" \
    --temperature ${TEMPERATURE} \
    --use_few_shot

echo ""
echo "=========================================="
echo "✅ 所有2种组合测试完成！"
echo "=========================================="
echo ""
echo "生成的文件:"
echo "  3. ${OUTPUT_DIR}/enhanced_test_dataset_predictions_finetuned_zero_shot.jsonl"
echo "  4. ${OUTPUT_DIR}/enhanced_test_dataset_predictions_finetuned_few_shot.jsonl"
echo ""
echo "对应的summary文件:"
echo "  3. ${OUTPUT_DIR}/enhanced_test_dataset_predictions_finetuned_zero_shot_summary.json"
echo "  4. ${OUTPUT_DIR}/enhanced_test_dataset_predictions_finetuned_few_shot_summary.json"

