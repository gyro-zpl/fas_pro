#!/bin/bash
#
# 检查微调环境是否准备就绪
#

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║      ANES群体态度微调 - 环境检查                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"

SUCCESS=0
FAIL=0

# 检查函数
check_file() {
    if [ -f "$1" ]; then
        echo "✅ $2"
        ((SUCCESS++))
    else
        echo "❌ $2 - 文件不存在: $1"
        ((FAIL++))
    fi
}

check_dir() {
    if [ -d "$1" ]; then
        echo "✅ $2"
        ((SUCCESS++))
    else
        echo "⚠️  $2 - 目录不存在: $1"
        ((FAIL++))
    fi
}

echo ""
echo "1️⃣  检查数据文件"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file "/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_train_enhanced.jsonl" "训练集"
check_file "/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_enhanced.jsonl" "测试集"
check_file "/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_metadata.csv" "元数据"

echo ""
echo "2️⃣  检查训练脚本"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file "/home/cyyuan/FAS_PRO/sft/finetune_anes_groups.py" "微调脚本"
check_file "/home/cyyuan/FAS_PRO/sft/inference_anes_groups.py" "推理脚本"
check_file "/home/cyyuan/FAS_PRO/sft/scripts/run_finetune_groups.sh" "运行脚本"
check_file "/home/cyyuan/FAS_PRO/sft/scripts/run_finetune_groups_quick.sh" "快速测试脚本"

echo ""
echo "3️⃣  检查文档"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
check_file "/home/cyyuan/FAS_PRO/data/anes/extend_features/STEP5_微调说明.md" "微调说明文档"
check_file "/home/cyyuan/FAS_PRO/data/anes/extend_features/快速使用指南.md" "快速使用指南"

echo ""
echo "4️⃣  检查输出目录"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
OUTPUT_DIR="/home/cyyuan/FAS_PRO/sft/sft_models"
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
    echo "✅ 创建输出目录: $OUTPUT_DIR"
    ((SUCCESS++))
else
    echo "✅ 输出目录存在"
    ((SUCCESS++))
fi

echo ""
echo "5️⃣  检查Python依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查Python包
check_package() {
    python3 -c "import $1" 2>/dev/null
    if [ $? -eq 0 ]; then
        VERSION=$(python3 -c "import $1; print($1.__version__)" 2>/dev/null)
        echo "✅ $2 (版本: $VERSION)"
        ((SUCCESS++))
    else
        echo "❌ $2 - 未安装"
        ((FAIL++))
    fi
}

check_package "torch" "PyTorch"
check_package "transformers" "Transformers"
check_package "peft" "PEFT (LoRA)"
check_package "datasets" "Datasets"

echo ""
echo "6️⃣  检查GPU"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    if [ $GPU_COUNT -gt 0 ]; then
        echo "✅ 检测到 $GPU_COUNT 个GPU"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | while read line; do
            echo "   $line"
        done
        ((SUCCESS++))
    else
        echo "⚠️  未检测到GPU"
        ((FAIL++))
    fi
else
    echo "⚠️  nvidia-smi 未找到"
    ((FAIL++))
fi

echo ""
echo "7️⃣  数据集统计"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TRAIN_SAMPLES=$(wc -l < /home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_train_enhanced.jsonl)
TEST_SAMPLES=$(wc -l < /home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_enhanced.jsonl)
echo "   训练集样本: $TRAIN_SAMPLES"
echo "   测试集样本: $TEST_SAMPLES"
echo "   总样本数:   $((TRAIN_SAMPLES + TEST_SAMPLES))"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "检查结果"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "   ✅ 成功: $SUCCESS 项"
echo "   ❌ 失败: $FAIL 项"
echo ""

if [ $FAIL -eq 0 ]; then
    echo "🎉 环境检查通过！可以开始训练。"
    echo ""
    echo "运行快速测试:"
    echo "  bash /home/cyyuan/FAS_PRO/sft/scripts/run_finetune_groups_quick.sh"
    echo ""
    echo "运行完整训练:"
    echo "  bash /home/cyyuan/FAS_PRO/sft/scripts/run_finetune_groups.sh"
else
    echo "⚠️  存在 $FAIL 个问题，请先解决。"
fi

echo ""

