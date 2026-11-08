#!/bin/bash
# 使用大规模训练集和优化参数重新训练

echo "================================"
echo "ANES候选人支持度预测 - 大规模优化训练"
echo "================================"
echo ""
echo "关键改进:"
echo "  ✅ 训练集: 2,491 → 7,790 样本 (3.1x)"
echo "  ✅ 学习率: 1e-4 → 3e-5 (更稳定)"
echo "  ✅ 有效batch: 16 → 32 (更好的梯度估计)"
echo "  ✅ 策略多样性: 5 → 9 种"
echo "  ✅ 候选人覆盖: 1 → 30 个"
echo ""
echo "预期效果:"
echo "  • 训练Loss更稳定下降"
echo "  • Eval Loss降低 (目标 <1.3)"
echo "  • 泛化性能提升"
echo ""
echo "开始训练..."
echo "================================"
echo ""

python /home/cyyuan/FAS_PRO/sft/finetune_anes_llama3.py \
  --train_file /home/cyyuan/FAS_PRO/data/anes/finetune/train_dataset_large.jsonl \
  --eval_file /home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl \
  --output_dir /home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora_large \
  --num_train_epochs 3 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 3e-5 \
  --save_steps 200 \
  --eval_steps 200 \
  --logging_steps 50 \
  --seed 42

echo ""
echo "================================"
echo "训练完成！"
echo "================================"
echo ""
echo "输出目录: /home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora_large"
echo ""
echo "查看训练日志:"
echo "  python /home/cyyuan/FAS_PRO/sft/view_training_log.py \\"
echo "    /home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora_large/checkpoint-XXX"
echo ""
echo "注意: 如果想使用warmup，需要手动在finetune_anes_llama3.py中添加参数"
echo ""
