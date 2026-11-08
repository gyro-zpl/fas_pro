#!/bin/bash
#
# ANES群体态度预测 - Llama 3.1 LoRA微调
# 运行脚本
#

echo "================================"
echo "ANES群体态度预测 - 开始微调"
echo "================================"

cd /home/cyyuan/FAS_PRO/sft

export CUDA_VISIBLE_DEVICES=2
# 默认参数
TRAIN_FILE="/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_train_enhanced.jsonl"
EVAL_FILE="/home/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_enhanced.jsonl"
OUTPUT_DIR="/home/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora"
MODEL_ID="meta-llama/Llama-3.1-8B-Instruct"

# 训练参数
NUM_EPOCHS=3
BATCH_SIZE=4
GRAD_ACC=4
LR=1e-4
MAX_LENGTH=512

# LoRA参数
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1

echo ""
echo "配置参数:"
echo "  模型: $MODEL_ID"
echo "  训练数据: $TRAIN_FILE"
echo "  测试数据: $EVAL_FILE"
echo "  输出目录: $OUTPUT_DIR"
echo "  训练轮数: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE × $GRAD_ACC"
echo "  学习率: $LR"
echo "  LoRA Rank: $LORA_R"
echo ""
echo "开始训练..."
echo ""

python finetune_anes_groups.py \
    --train_file "$TRAIN_FILE" \
    --eval_file "$EVAL_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --model_id "$MODEL_ID" \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACC \
    --learning_rate $LR \
    --max_length $MAX_LENGTH \
    --lora_r $LORA_R \
    --lora_alpha $LORA_ALPHA \
    --lora_dropout $LORA_DROPOUT \
    --save_steps 100 \
    --eval_steps 100 \
    --logging_steps 10

echo ""
echo "================================"
echo "✅ 训练完成！"
echo "================================"
echo ""
echo "模型保存在: $OUTPUT_DIR/final_model"
echo ""
echo "运行推理评估:"
echo "  python inference_anes_groups.py --model_path $OUTPUT_DIR/final_model"

