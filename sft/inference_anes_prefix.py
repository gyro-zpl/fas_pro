#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES选民支持度预测 - Prefix Tuning模型推理脚本

使用方法：
python inference_anes_prefix.py --model_path ./sft_models/anes_llama3_prefix/final_model \
                                 --test_file test_dataset.jsonl
"""

import torch
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from typing import List, Dict
import numpy as np
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ANES Prefix Tuning模型推理')
    
    parser.add_argument('--model_path', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_prefix/final_model',
                       help='Prefix Tuning模型路径')
    parser.add_argument('--base_model', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='基础模型ID')
    parser.add_argument('--test_file', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='测试数据文件')
    parser.add_argument('--output_file', type=str,
                       default='predictions_prefix.jsonl',
                       help='预测结果输出文件')
    parser.add_argument('--max_length', type=int,
                       default=512,
                       help='最大输入长度')
    parser.add_argument('--max_new_tokens', type=int,
                       default=10,
                       help='最大生成token数')
    parser.add_argument('--batch_size', type=int,
                       default=1,
                       help='批处理大小')
    
    return parser.parse_args()


def load_jsonl_data(file_path: str) -> List[Dict]:
    """加载JSONL格式数据"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


def create_prompt(example: Dict) -> str:
    """创建推理提示"""
    system_prompt = (
        "You are an expert in political science specializing in voter behavior prediction. "
        "Your task is to predict a voter's support score (0-100) for a political candidate "
        "based on the voter's demographics, political views, and the candidate's characteristics. "
        "Respond with ONLY a number between 0 and 100."
    )
    
    user_message = example['instruction'] + example['input']
    
    # Llama 3.1对话格式
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt


def extract_score(text: str) -> float:
    """从模型输出中提取分数"""
    try:
        # 尝试直接转换为数字
        score = float(text.strip())
        # 限制在0-100范围
        return max(0, min(100, score))
    except:
        # 尝试提取第一个数字
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            score = float(numbers[0])
            return max(0, min(100, score))
        # 如果无法解析，返回50
        return 50.0


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES Prefix Tuning模型推理")
    print("="*80)
    print(f"模型路径: {args.model_path}")
    print(f"测试文件: {args.test_file}")
    print("="*80)
    
    # 1. 加载模型和tokenizer
    print("\n步骤1: 加载模型")
    print(f"  加载基础模型: {args.base_model}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        use_fast=False,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16
    )
    
    print(f"  加载Prefix Tuning权重: {args.model_path}")
    model = PeftModel.from_pretrained(base_model, args.model_path)
    model.eval()
    
    print("  ✅ 模型加载完成")
    
    # 显示模型信息
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n  模型信息:")
    print(f"    总参数: {total_params:,}")
    print(f"    Prefix参数: {trainable_params:,}")
    print(f"    Prefix占比: {trainable_params/total_params*100:.4f}%")
    
    # 2. 加载测试数据
    print("\n步骤2: 加载测试数据")
    test_data = load_jsonl_data(args.test_file)
    print(f"  测试样本数: {len(test_data)}")
    
    # 3. 推理
    print("\n步骤3: 开始推理")
    predictions = []
    true_scores = []
    
    for example in tqdm(test_data, desc="推理进度"):
        # 创建提示
        prompt = create_prompt(example)
        
        # Tokenize
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            max_length=args.max_length,
            truncation=True
        ).to(model.device)
        
        # 生成
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,  # 贪婪解码
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        # 解码
        generated_text = tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        # 提取分数
        pred_score = extract_score(generated_text)
        true_score = example['metadata']['true_score']
        
        predictions.append({
            'input': example['input'][:100] + '...',  # 截断输入用于展示
            'true_score': true_score,
            'predicted_score': pred_score,
            'generated_text': generated_text.strip(),
            'strategy': example['metadata'].get('strategy', 'unknown'),
            'error': abs(true_score - pred_score)
        })
        
        true_scores.append(true_score)
    
    print("\n  ✅ 推理完成")
    
    # 4. 计算评估指标
    print("\n步骤4: 计算评估指标")
    pred_scores = [p['predicted_score'] for p in predictions]
    
    # MAE
    mae = mean_absolute_error(true_scores, pred_scores)
    
    # 分类准确率
    def score_to_bin(score):
        if score < 20: return 0
        elif score < 40: return 1
        elif score < 60: return 2
        elif score < 80: return 3
        else: return 4
    
    true_bins = [score_to_bin(s) for s in true_scores]
    pred_bins = [score_to_bin(s) for s in pred_scores]
    accuracy = accuracy_score(true_bins, pred_bins)
    
    # 按策略分层评估
    strategy_maes = {}
    for pred in predictions:
        strategy = pred['strategy']
        if strategy not in strategy_maes:
            strategy_maes[strategy] = []
        strategy_maes[strategy].append(pred['error'])
    
    # 5. 输出结果
    print("\n" + "="*80)
    print("📊 评估结果")
    print("="*80)
    print(f"整体MAE: {mae:.2f}")
    print(f"分类准确率（5类）: {accuracy:.2%}")
    print(f"\n按策略分层MAE:")
    for strategy, errors in sorted(strategy_maes.items()):
        print(f"  {strategy:25s}: MAE = {np.mean(errors):.2f} (n={len(errors)})")
    print("="*80)
    
    # 6. 保存预测结果
    output_path = Path(args.output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 预测结果已保存到: {output_path}")
    
    # 显示一些示例
    print("\n" + "="*80)
    print("📝 预测示例（前5个）")
    print("="*80)
    for i, pred in enumerate(predictions[:5], 1):
        print(f"\n示例 {i}:")
        print(f"  输入: {pred['input']}")
        print(f"  真实分数: {pred['true_score']:.1f}")
        print(f"  预测分数: {pred['predicted_score']:.1f}")
        print(f"  误差: {pred['error']:.1f}")
        print(f"  策略: {pred['strategy']}")
        print(f"  生成文本: '{pred['generated_text']}'")
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print("="*80)


if __name__ == "__main__":
    main()

