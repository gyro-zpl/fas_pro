#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES选民支持度预测任务 - Llama 3.1 Prefix Tuning微调脚本

功能：
1. 加载ANES训练/测试数据集
2. 使用Prefix Tuning微调Llama 3.1模型
3. 支持训练和评估
4. 计算MAE等评估指标

Prefix Tuning原理：
在输入序列前添加可学习的"虚拟tokens"（prefix），这些prefix通过训练学习任务特定知识，
而模型参数保持冻结。相比LoRA，Prefix Tuning参数更少，更适合生成任务。

使用方法：
python finetune_anes_prefix.py --train_file train_dataset.jsonl --eval_file test_dataset.jsonl
"""

from datasets import Dataset
import pandas as pd
import torch
import numpy as np
import json
from pathlib import Path
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    DataCollatorForSeq2Seq, 
    TrainingArguments, 
    Trainer,
    TrainerCallback
)
from peft import PrefixTuningConfig, TaskType, get_peft_model
import argparse
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, accuracy_score


class ANESEvaluationCallback(TrainerCallback):
    """自定义评估回调，计算MAE等指标"""
    
    def __init__(self, eval_dataset, tokenizer, model):
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.model = model
        self.best_mae = float('inf')
    
    def on_evaluate(self, args, state, control, **kwargs):
        """在评估时计算自定义指标"""
        print("\n" + "="*80)
        print("自定义评估...")
        print("="*80)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ANES Llama 3.1 Prefix Tuning微调')
    
    # 数据参数
    parser.add_argument('--train_file', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/train_dataset_fixed.jsonl',
                       help='训练数据文件路径（排除2024年数据）')
    parser.add_argument('--eval_file', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='评估数据文件路径')
    
    # 模型参数
    parser.add_argument('--model_id', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='预训练模型ID')
    parser.add_argument('--max_length', type=int,
                       default=512,
                       help='最大序列长度（ANES prompt较长，建议512）')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_prefix',
                       help='输出目录')
    parser.add_argument('--num_train_epochs', type=int,
                       default=3,
                       help='训练轮数')
    parser.add_argument('--per_device_train_batch_size', type=int,
                       default=4,
                       help='每设备训练batch size')
    parser.add_argument('--per_device_eval_batch_size', type=int,
                       default=4,
                       help='每设备评估batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int,
                       default=4,
                       help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float,
                       default=1e-3,
                       help='学习率（Prefix Tuning通常用较大学习率）')
    parser.add_argument('--save_steps', type=int,
                       default=100,
                       help='保存检查点的步数')
    parser.add_argument('--eval_steps', type=int,
                       default=100,
                       help='评估的步数')
    parser.add_argument('--logging_steps', type=int,
                       default=10,
                       help='日志记录步数')
    
    # Prefix Tuning参数
    parser.add_argument('--num_virtual_tokens', type=int,
                       default=20,
                       help='虚拟token数量（prefix长度），推荐10-30')
    parser.add_argument('--prefix_projection', type=bool,
                       default=True,
                       help='是否使用MLP重参数化（提高表达能力）')
    parser.add_argument('--encoder_hidden_size', type=int,
                       default=512,
                       help='如果使用prefix_projection，MLP的隐藏层大小')
    
    # 其他
    parser.add_argument('--seed', type=int,
                       default=42,
                       help='随机种子')
    
    return parser.parse_args()


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process_func(example, tokenizer, max_length):
    """
    处理单个样本，转换为模型输入格式
    
    ANES任务特定的System Prompt：
    指导模型这是一个选民支持度预测任务
    """
    input_ids, attention_mask, labels = [], [], []
    
    # ANES任务的System Prompt
    system_prompt = (
        "You are an expert in political science specializing in voter behavior prediction. "
        "Your task is to predict a voter's support score (0-100) for a political candidate "
        "based on the voter's demographics, political views, and the candidate's characteristics. "
        "Respond with ONLY a number between 0 and 100."
    )
    
    # 构建完整的instruction
    user_message = example['instruction'] + example['input']
    
    # Llama 3.1的对话格式
    instruction = tokenizer(
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n",
        add_special_tokens=False
    )
    
    response = tokenizer(
        f"{example['output']}<|eot_id|>",
        add_special_tokens=False
    )
    
    # 组合input_ids
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    # Labels: instruction部分设为-100（不计算损失），response部分计算损失
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [tokenizer.pad_token_id]
    
    # 截断
    if len(input_ids) > max_length:
        input_ids = input_ids[:max_length]
        attention_mask = attention_mask[:max_length]
        labels = labels[:max_length]
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def compute_metrics(eval_dataset, predictions, tokenizer):
    """
    计算ANES任务的评估指标
    
    指标：
    1. MAE (Mean Absolute Error)
    2. 分类准确率（5类）
    3. 按策略分层的MAE
    """
    # 解析预测结果
    pred_scores = []
    true_scores = []
    
    for pred_text, sample in zip(predictions, eval_dataset):
        # 从预测文本中提取数字
        try:
            # 尝试直接转换为数字
            pred_score = float(pred_text.strip())
            # 限制在0-100范围
            pred_score = max(0, min(100, pred_score))
        except:
            # 如果无法解析，使用默认值50
            pred_score = 50.0
        
        pred_scores.append(pred_score)
        true_scores.append(sample['metadata']['true_score'])
    
    # 计算MAE
    mae = mean_absolute_error(true_scores, pred_scores)
    
    # 计算分类准确率
    def score_to_bin(score):
        if score < 20: return 0  # 强烈反对
        elif score < 40: return 1  # 反对
        elif score < 60: return 2  # 中立
        elif score < 80: return 3  # 支持
        else: return 4  # 强烈支持
    
    true_bins = [score_to_bin(s) for s in true_scores]
    pred_bins = [score_to_bin(s) for s in pred_scores]
    accuracy = accuracy_score(true_bins, pred_bins)
    
    # 按策略分层评估
    strategy_maes = {}
    for sample, true_score, pred_score in zip(eval_dataset, true_scores, pred_scores):
        strategy = sample['metadata'].get('strategy', 'unknown')
        if strategy not in strategy_maes:
            strategy_maes[strategy] = []
        strategy_maes[strategy].append(abs(true_score - pred_score))
    
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"整体MAE: {mae:.2f}")
    print(f"分类准确率（5类）: {accuracy:.2%}")
    print(f"\n按策略分层MAE:")
    for strategy, errors in sorted(strategy_maes.items()):
        print(f"  {strategy:25s}: MAE = {np.mean(errors):.2f} (n={len(errors)})")
    print("="*80)
    
    return {
        'mae': mae,
        'accuracy': accuracy,
        'strategy_maes': {k: np.mean(v) for k, v in strategy_maes.items()}
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES选民支持度预测 - Llama 3.1 Prefix Tuning微调")
    print("="*80)
    print(f"训练文件: {args.train_file}")
    print(f"评估文件: {args.eval_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"模型ID: {args.model_id}")
    print("="*80)
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载数据
    print("\n步骤1: 加载数据")
    train_data = load_jsonl_data(args.train_file)
    eval_data = load_jsonl_data(args.eval_file)
    print(f"  训练样本: {len(train_data)}")
    print(f"  评估样本: {len(eval_data)}")
    
    # 数据统计
    train_strategies = {}
    for d in train_data:
        strategy = d['metadata'].get('strategy', 'unknown')
        train_strategies[strategy] = train_strategies.get(strategy, 0) + 1
    print(f"\n  训练集策略分布:")
    for strategy, count in sorted(train_strategies.items(), key=lambda x: -x[1]):
        print(f"    {strategy:25s}: {count:4d} ({count/len(train_data)*100:5.1f}%)")
    
    # 2. 加载模型和tokenizer
    print("\n步骤2: 加载模型和Tokenizer")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    model.enable_input_require_grads()  # 开启梯度检查点时需要
    
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    
    # 3. 准备数据集
    print("\n步骤3: 处理数据集")
    df_train = pd.DataFrame(train_data)
    df_eval = pd.DataFrame(eval_data)
    
    ds_train = Dataset.from_pandas(df_train)
    ds_eval = Dataset.from_pandas(df_eval)
    
    # 应用处理函数
    tokenized_train = ds_train.map(
        lambda x: process_func(x, tokenizer, args.max_length),
        remove_columns=ds_train.column_names
    )
    tokenized_eval = ds_eval.map(
        lambda x: process_func(x, tokenizer, args.max_length),
        remove_columns=ds_eval.column_names
    )
    
    print(f"  训练集处理完成: {len(tokenized_train)} 样本")
    print(f"  评估集处理完成: {len(tokenized_eval)} 样本")
    
    # 4. 配置Prefix Tuning
    print("\n步骤4: 配置Prefix Tuning")
    print("\n💡 Prefix Tuning原理:")
    print("   在每层Transformer的输入前添加可学习的连续向量（prefix）")
    print("   这些prefix相当于'软提示'，通过训练学习任务特定知识")
    print("   相比LoRA，参数更少，特别适合生成任务\n")
    
    prefix_config = PrefixTuningConfig(
        task_type=TaskType.CAUSAL_LM,
        num_virtual_tokens=args.num_virtual_tokens,
        prefix_projection=args.prefix_projection,
        encoder_hidden_size=args.encoder_hidden_size if args.prefix_projection else None,
        inference_mode=False,  # 训练模式
    )
    
    model = get_peft_model(model, prefix_config)
    print("  Prefix Tuning配置:")
    print(f"    虚拟Token数量: {args.num_virtual_tokens}")
    print(f"    使用MLP重参数化: {args.prefix_projection}")
    if args.prefix_projection:
        print(f"    MLP隐藏层大小: {args.encoder_hidden_size}")
    print("\n  可训练参数:")
    model.print_trainable_parameters()
    
    # 计算参数量对比
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  参数效率分析:")
    print(f"    总参数: {total_params:,}")
    print(f"    可训练参数: {trainable_params:,}")
    print(f"    训练比例: {trainable_params/total_params*100:.4f}%")
    print(f"    参数节省: {(1-trainable_params/total_params)*100:.2f}%")
    
    # 5. 训练配置
    print("\n步骤5: 配置训练参数")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,  # Prefix Tuning通常用较大学习率
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,  # 只保留最近3个检查点
        eval_strategy="steps",
        save_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        save_on_each_node=True,
        gradient_checkpointing=True,
        fp16=False,
        bf16=True,
        seed=args.seed,
        report_to="none",  # 不使用wandb等
        remove_unused_columns=False,
        warmup_steps=100,  # Prefix Tuning建议使用warmup
    )
    
    print(f"  训练轮数: {args.num_train_epochs}")
    print(f"  学习率: {args.learning_rate} (注：Prefix Tuning通常用较大学习率)")
    print(f"  Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} (累积)")
    print(f"  有效batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"  Warmup steps: {training_args.warmup_steps}")
    
    # 6. 创建Trainer
    print("\n步骤6: 创建Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    )
    
    # 7. 开始训练
    print("\n步骤7: 开始训练")
    print("="*80)
    trainer.train()
    
    # 8. 保存最终模型
    print("\n步骤8: 保存最终模型")
    final_model_path = f"{args.output_dir}/final_model"
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"  最终模型已保存到: {final_model_path}")
    print(f"  💡 Prefix Tuning模型非常小（通常< 10MB）！")
    
    # 9. 评估
    print("\n步骤9: 最终评估")
    print("="*80)
    
    # 简单评估（使用trainer）
    eval_results = trainer.evaluate()
    print("\nTrainer评估结果:")
    for key, value in eval_results.items():
        print(f"  {key}: {value:.4f}")
    
    # 保存评估结果
    results_file = f"{args.output_dir}/eval_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(eval_results, f, indent=2)
    print(f"\n评估结果已保存到: {results_file}")
    
    # 保存配置信息
    config_file = f"{args.output_dir}/training_config.json"
    config_info = {
        'method': 'Prefix Tuning',
        'num_virtual_tokens': args.num_virtual_tokens,
        'prefix_projection': args.prefix_projection,
        'encoder_hidden_size': args.encoder_hidden_size if args.prefix_projection else None,
        'learning_rate': args.learning_rate,
        'num_train_epochs': args.num_train_epochs,
        'trainable_params': trainable_params,
        'trainable_percentage': f"{trainable_params/total_params*100:.4f}%"
    }
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_info, f, indent=2, ensure_ascii=False)
    print(f"训练配置已保存到: {config_file}")
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - 最终模型: {final_model_path}")
    print(f"  - 评估结果: {results_file}")
    print(f"  - 训练配置: {config_file}")
    print(f"  - 训练日志: {args.output_dir}/runs/")
    
    print(f"\n📊 Prefix Tuning vs LoRA 对比:")
    print(f"  Prefix Tuning特点:")
    print(f"    ✅ 参数更少（通常是LoRA的1/10）")
    print(f"    ✅ 模型文件极小（几MB vs 几十MB）")
    print(f"    ✅ 特别适合生成任务")
    print(f"    ⚠️  训练可能需要更多轮次")
    print(f"    ⚠️  对超参数更敏感")
    
    print(f"\n使用模型进行推理:")
    print(f"  python inference_anes_prefix.py --model_path {final_model_path}")


if __name__ == "__main__":
    main()

