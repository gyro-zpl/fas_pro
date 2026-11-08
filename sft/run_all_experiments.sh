#!/bin/bash

# 运行所有实验的主脚本
# 包括候选人支持度预测和群体态度预测的4种组合

set -e  # 遇到错误立即退出

LOG_DIR="/home/cyyuan/FAS_PRO/sft/logs"
mkdir -p "${LOG_DIR}"

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/all_experiments_${TIMESTAMP}.log"

echo "=========================================="
echo "开始运行所有实验"
echo "日志文件: ${LOG_FILE}"
echo "开始时间: $(date)"
echo "=========================================="

# 记录到日志文件
exec > >(tee -a "${LOG_FILE}") 2>&1

echo ""
echo "🔵 [阶段1/2] 候选人支持度预测 - 4种组合"
echo "=========================================="
bash /home/cyyuan/FAS_PRO/sft/run_all_4_combinations.sh

echo ""
echo "🟢 [阶段2/2] 群体态度预测 - 4种组合"
echo "=========================================="
bash /home/cyyuan/FAS_PRO/sft/run_all_4_combinations_groups.sh

echo ""
echo "=========================================="
echo "✅ 所有实验完成！"
echo "结束时间: $(date)"
echo "=========================================="
echo ""
echo "生成的文件位置："
echo "  候选人预测: /home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/"
echo "  群体预测: /home/cyyuan/FAS_PRO/data/anes/extend_features/results/"
echo ""
echo "日志文件: ${LOG_FILE}"


