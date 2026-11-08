#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES群体态度预测任务 - Llama 3.1微调脚本

功能：
1. 加载ANES群体态度训练/测试数据集
2. 使用LoRA微调Llama 3.1模型
3. 支持训练和评估
4. 计算MAE、RMSE等评估指标

使用方法：
python finetune_anes_groups.py --train_file train_dataset.jsonl --eval_file test_dataset.jsonl
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
from peft import LoraConfig, TaskType, get_peft_model
import argparse
from typing import Dict, List
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score


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
    parser = argparse.ArgumentParser(description='ANES群体态度 Llama 3.1 微调')
    
    # 数据参数
    parser.add_argument('--train_file', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/extend_features/data/group_train_full.jsonl',
                       help='训练数据文件路径')
    parser.add_argument('--eval_file', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_balanced_v3.jsonl',
                       help='评估数据文件路径')
    
    # 模型参数
    parser.add_argument('--model_id', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='预训练模型ID')
    parser.add_argument('--max_length', type=int,
                       default=512,
                       help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora',
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
                       default=1e-4,
                       help='学习率')
    parser.add_argument('--warmup_steps', type=int,
                       default=0,
                       help='学习率warmup步数')
    parser.add_argument('--save_steps', type=int,
                       default=100,
                       help='保存检查点的步数')
    parser.add_argument('--eval_steps', type=int,
                       default=100,
                       help='评估的步数')
    parser.add_argument('--logging_steps', type=int,
                       default=10,
                       help='日志记录步数')
    
    # LoRA参数
    parser.add_argument('--lora_r', type=int,
                       default=8,
                       help='LoRA秩')
    parser.add_argument('--lora_alpha', type=int,
                       default=32,
                       help='LoRA alpha')
    parser.add_argument('--lora_dropout', type=float,
                       default=0.1,
                       help='LoRA dropout')
    
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
    
    群体态度任务特定的System Prompt：
    指导模型这是一个群体态度预测任务
    """
    input_ids, attention_mask, labels = [], [], []
    
    # 群体态度任务的System Prompt
    system_prompt = (
        "You are an expert in political science and social psychology specializing in group attitudes. "
        "Your task is to predict a voter's feeling thermometer score (0-100) toward a social group "
        "based on the voter's demographics, political views, and the group's characteristics. "
        "A score of 0 means very cold/unfavorable feelings, 100 means very warm/favorable feelings, "
        "and 50 means neutral. Respond with ONLY a number between 0 and 100."
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
    计算群体态度任务的评估指标
    
    指标：
    1. MAE (Mean Absolute Error)
    2. RMSE (Root Mean Squared Error)
    3. 自定义准确率（同一20分档或误差≤10）
    4. 按策略分层的MAE
    5. 按群体分层的MAE
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
    
    # 计算MAE和RMSE
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = np.sqrt(mean_squared_error(true_scores, pred_scores))
    
    # 计算自定义准确率：同一20分档 OR 误差≤10
    def score_to_bin(score):
        """0-100分数转为20分档（0-4）"""
        return int(score // 20)
    
    correct = 0
    for true_score, pred_score in zip(true_scores, pred_scores):
        # 条件1：在同一20分档
        same_bin = (score_to_bin(true_score) == score_to_bin(pred_score))
        # 条件2：误差≤10
        small_error = (abs(true_score - pred_score) <= 10)
        
        if same_bin or small_error:
            correct += 1
    
    custom_accuracy = correct / len(true_scores)
    
    # 按策略分层评估
    strategy_metrics = {}
    for sample, true_score, pred_score in zip(eval_dataset, true_scores, pred_scores):
        strategy = sample['metadata'].get('strategy', 'unknown')
        if strategy not in strategy_metrics:
            strategy_metrics[strategy] = {'errors': [], 'true': [], 'pred': []}
        strategy_metrics[strategy]['errors'].append(abs(true_score - pred_score))
        strategy_metrics[strategy]['true'].append(true_score)
        strategy_metrics[strategy]['pred'].append(pred_score)
    
    # 按群体分层评估
    group_metrics = {}
    for sample, true_score, pred_score in zip(eval_dataset, true_scores, pred_scores):
        group = sample['metadata'].get('group', 'unknown')
        if group not in group_metrics:
            group_metrics[group] = {'errors': [], 'true': [], 'pred': []}
        group_metrics[group]['errors'].append(abs(true_score - pred_score))
        group_metrics[group]['true'].append(true_score)
        group_metrics[group]['pred'].append(pred_score)
    
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    print(f"整体指标:")
    print(f"  MAE:              {mae:.2f}")
    print(f"  RMSE:             {rmse:.2f}")
    print(f"  自定义准确率:      {custom_accuracy:.2%}")
    print(f"  样本数:           {len(true_scores)}")
    
    print(f"\n按策略分层MAE:")
    for strategy, metrics in sorted(strategy_metrics.items()):
        strategy_mae = np.mean(metrics['errors'])
        print(f"  {strategy:25s}: MAE = {strategy_mae:.2f} (n={len(metrics['errors'])})")
    
    print(f"\n按群体分层MAE:")
    for group, metrics in sorted(group_metrics.items()):
        group_mae = np.mean(metrics['errors'])
        print(f"  {group:30s}: MAE = {group_mae:.2f} (n={len(metrics['errors'])})")
    
    print("="*80)
    
    return {
        'mae': mae,
        'rmse': rmse,
        'custom_accuracy': custom_accuracy,
        'strategy_maes': {k: np.mean(v['errors']) for k, v in strategy_metrics.items()},
        'group_maes': {k: np.mean(v['errors']) for k, v in group_metrics.items()}
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES群体态度预测 - Llama 3.1 LoRA微调")
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
    print(f"\n  训练集策略分布:")
    train_strategies = {}
    for d in train_data:
        strategy = d['metadata'].get('strategy', 'unknown')
        train_strategies[strategy] = train_strategies.get(strategy, 0) + 1
    for strategy, count in sorted(train_strategies.items(), key=lambda x: -x[1]):
        print(f"    {strategy:25s}: {count:4d} ({count/len(train_data)*100:5.1f}%)")
    
    print(f"\n  训练集群体分布:")
    train_groups = {}
    for d in train_data:
        group = d['metadata'].get('group', 'unknown')
        train_groups[group] = train_groups.get(group, 0) + 1
    for group, count in sorted(train_groups.items(), key=lambda x: -x[1]):
        print(f"    {group:30s}: {count:4d} ({count/len(train_data)*100:5.1f}%)")
    
    print(f"\n  测试集群体分布:")
    eval_groups = {}
    for d in eval_data:
        group = d['metadata'].get('group', 'unknown')
        eval_groups[group] = eval_groups.get(group, 0) + 1
    for group, count in sorted(eval_groups.items()):
        print(f"    {group:30s}: {count:4d} ({count/len(eval_data)*100:5.1f}%)")
    
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
    
    # 4. 配置LoRA
    print("\n步骤4: 配置LoRA")
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout
    )
    
    model = get_peft_model(model, lora_config)
    print("\n  LoRA配置:")
    print(f"    Rank (r): {args.lora_r}")
    print(f"    Alpha: {args.lora_alpha}")
    print(f"    Dropout: {args.lora_dropout}")
    print(f"    Target modules: {lora_config.target_modules}")
    print("\n  可训练参数:")
    model.print_trainable_parameters()
    
    # 5. 训练配置
    print("\n步骤5: 配置训练参数")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        save_total_limit=3,  # 只保留最近3个检查点
        eval_strategy="steps",  # 新版本transformers使用eval_strategy
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
    )
    
    print(f"  训练轮数: {args.num_train_epochs}")
    print(f"  学习率: {args.learning_rate}")
    print(f"  Warmup步数: {args.warmup_steps}")
    print(f"  Batch size: {args.per_device_train_batch_size} × {args.gradient_accumulation_steps} (累积)")
    print(f"  有效batch size: {args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    
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
    
    print("\n" + "="*80)
    print("✅ 训练完成！")
    print("="*80)
    print(f"\n输出文件:")
    print(f"  - 最终模型: {final_model_path}")
    print(f"  - 评估结果: {results_file}")
    print(f"  - 训练日志: {args.output_dir}/runs/")
    
    print(f"\n使用模型进行推理:")
    print(f"  python inference_anes_groups.py --model_path {final_model_path}")
    
    print(f"\n下一步建议:")
    print(f"  1. 使用inference脚本进行详细评估")
    print(f"  2. 或合并候选人+群体数据集进行多任务学习")


if __name__ == "__main__":
    main()

