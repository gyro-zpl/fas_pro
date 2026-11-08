#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES模型推理脚本

使用训练好的Llama 3.1 LoRA模型进行选民支持度预测

使用方法：
/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/finetune/scripts/inference.sh
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, accuracy_score
from tqdm import tqdm
import sys


def custom_accuracy(true_scores, pred_scores):
    """
    自定义准确率计算
    
    判断规则：
    1. 预测值和真实值在同一个20分段内（0-20, 20-40, 40-60, 60-80, 80-100）
    2. 或者预测值与真实值的绝对误差 ≤ 10
    
    满足任一条件即认为预测正确
    
    Args:
        true_scores: 真实分数列表
        pred_scores: 预测分数列表
    
    Returns:
        accuracy: 准确率 (0-1)
    """
    correct = 0
    total = len(true_scores)
    
    for true_val, pred_val in zip(true_scores, pred_scores):
        # 规则1：计算所在的20分段 (0-20=0, 20-40=1, 40-60=2, 60-80=3, 80-100=4)
        true_bin = min(int(true_val // 20), 4)  # 确保100分也在第4段
        pred_bin = min(int(pred_val // 20), 4)
        
        # 规则2：计算绝对误差
        abs_error = abs(true_val - pred_val)
        
        # 满足任一条件即正确
        if true_bin == pred_bin or abs_error <= 10:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ANES模型推理')
    
    parser.add_argument('--base_model', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='基础模型路径（Llama 3.1）')
    parser.add_argument('--lora_path', type=str,
                        default='/home/jpzhao_team/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/final_model',
                        help='LoRA适配器路径（训练好的checkpoint）')
    parser.add_argument('--test_file', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='测试数据文件')
    parser.add_argument('--output_file', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/enhanced_test_dataset_predictions_few_shot.jsonl',
                       help='预测结果输出文件（JSONL格式）')
    parser.add_argument('--max_new_tokens', type=int,
                       default=10,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float,
                       default=0.3,
                       help='生成温度（越低越确定）')
    parser.add_argument('--num_samples', type=int,
                       default=None,
                       help='测试样本数（None表示全部）')
    parser.add_argument('--use_few_shot', action='store_true',
                       help='是否使用few-shot examples（默认zero-shot）')
    parser.add_argument('--base_model_only', action='store_true',
                       help='仅使用base model，不加载LoRA（用于测试base model性能）')
    
    return parser.parse_args()


def load_model(base_model_path: str, lora_path: str = None, base_only: bool = False):
    """加载基础模型和LoRA适配器
    
    Args:
        base_model_path: Llama 3.1 基础模型路径
        lora_path: LoRA适配器checkpoint路径（base_only=True时可为None）
        base_only: 是否仅加载base model，不加载LoRA
    """
    print(f"加载基础模型: {base_model_path}")
    if not base_only and lora_path:
        print(f"加载LoRA适配器: {lora_path}")
    else:
        print("仅使用基础模型（不加载LoRA）")
    
    # 1. 加载 tokenizer（从base model）
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    # 设置 padding token（Llama 模型需要）
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # 2. 加载基础模型
    print("  加载基础模型...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # 3. 如果不是base_only，加载并合并 LoRA 适配器
    if not base_only and lora_path:
        print("  加载LoRA适配器...")
        model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.bfloat16
        )
        
        # 4. 合并权重以提高推理速度（可选）
        print("  合并LoRA权重...")
        model = model.merge_and_unload()
    
    print("  模型加载完成！")
    return model, tokenizer


def create_prompt(sample: dict, use_few_shot: bool = False) -> str:
    """创建推理prompt
    
    Args:
        sample: 测试样本
        use_few_shot: 是否使用few-shot examples
    """
    system_prompt = (
        "You are an expert in political science specializing in voter behavior prediction. "
        "Your task is to predict a voter's support score (0-100) for a political candidate "
        "based on the voter's demographics, political views, and the candidate's characteristics. "
        "Respond with ONLY a number between 0 and 100."
    )
    
    few_shot_prompt = (
        "Here are 5 examples covering different score ranges:\n\n"
        
        "Example 1 - Very High Support (80-100):\n"
        "Voter: 38-year-old Female White, Some high school\n"
        "Political: Strong Republican, Conservative\n"
        "Candidate: Trump (Republican, President)\n"
        "Party alignment: Same party, Year: 2024\n"
        "Predicted score: 92\n"
        "Reasoning: Strong partisan alignment with same-party candidate.\n\n"
        
        "Example 2 - Moderate-High Support (60-79):\n"
        "Voter: 38-year-old Male White, Some high school\n"
        "Political: Strong Republican, Moderate\n"
        "Candidate: Yang (Democrat, Candidate)\n"
        "Party alignment: Different party, Year: 2020\n"
        "Predicted score: 60\n"
        "Reasoning: Republican voter with moderate ideology shows some openness to Democrat candidate.\n\n"
        
        "Example 3 - Neutral/Mixed (40-59):\n"
        "Voter: 43-year-old Male, Some high school\n"
        "Political: Independent, Moderate\n"
        "Candidate: Buttigieg (Democrat, Candidate)\n"
        "Party alignment: Independent, Year: 2020\n"
        "Predicted score: 40\n"
        "Reasoning: Independent voters show moderate, context-dependent support.\n\n"
        
        "Example 4 - Moderate-Low Support (20-39):\n"
        "Voter: 38-year-old Female Hispanic, Some high school\n"
        "Political: Lean Democrat, Liberal\n"
        "Candidate: Trump (Republican, President)\n"
        "Party alignment: Different party, Year: 2024\n"
        "Predicted score: 20\n"
        "Reasoning: Lean Democrat with liberal ideology opposes Republican candidate.\n\n"
        
        "Example 5 - Very Low Support (0-19):\n"
        "Voter: 62-year-old Male White, Some college\n"
        "Political: Strong Democrat, Liberal\n"
        "Candidate: Trump (Republican, President)\n"
        "Party alignment: Different party, Year: 2024\n"
        "Predicted score: 2\n"
        "Reasoning: Strong partisan opposition with opposing ideology.\n\n"
        
        "Key patterns:\n"
        "- Score ranges: Very High (80-100), Moderate-High (60-79), Neutral (40-59), Moderate-Low (20-39), Very Low (0-19)\n"
        "- Party alignment is the strongest predictor\n"
        "- Ideology and partisan strength modulate the base prediction\n"
        "- Independents typically score in the middle range (30-70)\n"
        "- Consider year context and candidate characteristics\n\n"
    )
    
    background_prompt = """
    The data comes from American National Election Studies(ANES), 
    you need to consider many political and social factors, including the policies of the year and general approval ratings.
    """
    
    user_message = sample['instruction'] + sample['input'] + background_prompt
    
    # 根据use_few_shot参数决定是否包含few-shot examples
    if use_few_shot:
        system_content = f"{system_prompt}\n\n{few_shot_prompt}"
    else:
        system_content = system_prompt
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_message}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    return prompt


def predict(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float) -> str:
    """单个样本预测"""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=False,  # 贪心解码，更稳定
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
        )
    
    # 只取新生成的部分
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    return response

def extract_score(response: str) -> float:
    """从模型响应中提取分数"""
    try:
        # 尝试直接转换为数字
        score = float(response.strip())
        # 限制在0-100范围
        score = max(0, min(100, score))
        return score
    except:
        # 尝试从文本中提取数字
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            return max(0, min(100, score))
        # 无法解析，返回默认值
        return 50.0


def evaluate_predictions(predictions: list, test_data: list):
    """评估预测结果
    
    Args:
        predictions: 预测结果列表
        test_data: 测试数据列表
    """
    pred_scores = []
    true_scores = []
    
    for pred, sample in zip(predictions, test_data):
        pred_scores.append(pred['predicted_score'])
        true_scores.append(sample['metadata']['true_score'])
    
    # 模型评估
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = np.sqrt(np.mean((np.array(pred_scores) - np.array(true_scores))**2))
    accuracy = custom_accuracy(true_scores, pred_scores)
    
    # 按候选人分层评估
    candidate_data = {}
    for pred, sample in zip(predictions, test_data):
        candidate = sample['metadata'].get('candidate', 'unknown')
        if candidate not in candidate_data:
            candidate_data[candidate] = {'true': [], 'pred': []}
        candidate_data[candidate]['true'].append(sample['metadata']['true_score'])
        candidate_data[candidate]['pred'].append(pred['predicted_score'])
    
    # 计算每个候选人的指标
    candidate_results = {}
    for candidate, data in candidate_data.items():
        cand_mae = mean_absolute_error(data['true'], data['pred'])
        cand_rmse = np.sqrt(np.mean((np.array(data['pred']) - np.array(data['true']))**2))
        cand_acc = custom_accuracy(data['true'], data['pred'])
        
        candidate_results[candidate] = {
            'count': len(data['true']),
            'mae': float(cand_mae),
            'rmse': float(cand_rmse),
            'accuracy': float(cand_acc)
        }
    
    # 按策略分层评估
    strategy_data = {}
    for pred, sample in zip(predictions, test_data):
        strategy = sample['metadata'].get('strategy', 'unknown')
        if strategy not in strategy_data:
            strategy_data[strategy] = {'true': [], 'pred': []}
        strategy_data[strategy]['true'].append(sample['metadata']['true_score'])
        strategy_data[strategy]['pred'].append(pred['predicted_score'])
    
    # 计算每个策略的指标
    strategy_results = {}
    for strategy, data in strategy_data.items():
        strat_mae = mean_absolute_error(data['true'], data['pred'])
        strat_rmse = np.sqrt(np.mean((np.array(data['pred']) - np.array(data['true']))**2))
        strat_acc = custom_accuracy(data['true'], data['pred'])
        
        strategy_results[strategy] = {
            'count': len(data['true']),
            'mae': float(strat_mae),
            'rmse': float(strat_rmse),
            'accuracy': float(strat_acc)
        }
    
    # 按年份分层评估
    year_data = {}
    for pred, sample in zip(predictions, test_data):
        year = sample['metadata'].get('year', 'unknown')
        if year not in year_data:
            year_data[year] = {'true': [], 'pred': []}
        year_data[year]['true'].append(sample['metadata']['true_score'])
        year_data[year]['pred'].append(pred['predicted_score'])
    
    # 计算每个年份的指标
    year_results = {}
    for year, data in year_data.items():
        year_mae = mean_absolute_error(data['true'], data['pred'])
        year_rmse = np.sqrt(np.mean((np.array(data['pred']) - np.array(data['true']))**2))
        year_acc = custom_accuracy(data['true'], data['pred'])
        
        year_results[str(year)] = {
            'count': len(data['true']),
            'mae': float(year_mae),
            'rmse': float(year_rmse),
            'accuracy': float(year_acc)
        }
    
    return {
        'overall': {
            'mae': mae,
            'rmse': rmse,
            'accuracy': accuracy
        },
        'by_candidate': candidate_results,
        'by_strategy': strategy_results,
        'by_year': year_results
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES选民支持度预测 - 模型推理")
    print("="*80)
    print(f"基础模型: {args.base_model}")
    print(f"LoRA路径: {args.lora_path}")
    print(f"测试文件: {args.test_file}")
    print("="*80)
    
    # 1. 加载模型
    print("\n步骤1: 加载模型")
    print(f"  Few-shot: {'是' if args.use_few_shot else '否'}")
    print(f"  仅Base Model: {'是' if args.base_model_only else '否（使用Fine-tuned）'}")
    model, tokenizer = load_model(args.base_model, args.lora_path, base_only=args.base_model_only)
    
    # 2. 加载测试数据
    print("\n步骤2: 加载测试数据")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"  测试样本数: {len(test_data)}")
    
    # 3. 推理
    print("\n步骤3: 开始推理")
    predictions = []
    
    for i, sample in enumerate(tqdm(test_data, desc="预测进度")):
        # 创建prompt（使用few-shot或zero-shot）
        prompt = create_prompt(sample, use_few_shot=args.use_few_shot)
        
        # 主模型预测
        response = predict(model, tokenizer, prompt, args.max_new_tokens, args.temperature)
        pred_score = extract_score(response)
        
        # 准备结果字典
        result = {
            'sample_id': i,
            'candidate': sample['metadata'].get('candidate', 'unknown'),
            'year': sample['metadata'].get('year', 'unknown'),
            'true_score': sample['metadata']['true_score'],
            'predicted_score': pred_score,
            'raw_response': response,
            'error': abs(pred_score - sample['metadata']['true_score'])
        }
        
        predictions.append(result)
        #break
    print(f"  推理完成，共 {len(predictions)} 个样本")
    
    # 4. 评估
    print("\n步骤4: 评估结果")
    print("="*80)
    
    results = evaluate_predictions(predictions, test_data)
    
    # 显示模型结果
    model_type = "Base Model" if args.base_model_only else "Fine-tuned Model"
    prompt_type = "Few-shot" if args.use_few_shot else "Zero-shot"
    print(f"\n{model_type} ({prompt_type}) 评估结果:")
    print(f"  MAE: {results['overall']['mae']:.2f}")
    print(f"  RMSE: {results['overall']['rmse']:.2f}")
    print(f"  Accuracy: {results['overall']['accuracy']:.1%}")
    
    # 按候选人分层
    print(f"\n按候选人分层:")
    print(f"{'候选人':<15} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'样本数':>8}")
    print("-" * 55)
    for candidate in sorted(results['by_candidate'].keys()):
        metrics = results['by_candidate'][candidate]
        print(f"{candidate:<15} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} {metrics['accuracy']:>9.1%} {metrics['count']:>8}")
    
    # 按策略分层
    print(f"\n按策略分层:")
    print(f"{'策略':<25} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'样本数':>8}")
    print("-" * 65)
    for strategy in sorted(results['by_strategy'].keys()):
        metrics = results['by_strategy'][strategy]
        print(f"{strategy:<25} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} {metrics['accuracy']:>9.1%} {metrics['count']:>8}")
    
    # 按年份分层
    print(f"\n按年份分层:")
    print(f"{'年份':<10} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'样本数':>8}")
    print("-" * 50)
    for year in sorted(results['by_year'].keys()):
        metrics = results['by_year'][year]
        print(f"{year:<10} {metrics['mae']:>10.2f} {metrics['rmse']:>10.2f} {metrics['accuracy']:>9.1%} {metrics['count']:>8}")
    
    # 找出最好和最差的预测
    sorted_preds = sorted(predictions, key=lambda x: x['error'])
    
    print(f"\n最佳预测（top 3）:")
    for i, pred in enumerate(sorted_preds[:3], 1):
        print(f"  {i}. 候选人={pred['candidate']:10s} | "
              f"真实={pred['true_score']:5.1f} | "
              f"预测={pred['predicted_score']:5.1f} | "
              f"误差={pred['error']:5.2f}")
    
    print(f"\n最差预测（bottom 3）:")
    for i, pred in enumerate(sorted_preds[-3:], 1):
        print(f"  {i}. 候选人={pred['candidate']:10s} | "
              f"真实={pred['true_score']:5.1f} | "
              f"预测={pred['predicted_score']:5.1f} | "
              f"误差={pred['error']:5.2f}")
    
    # 5. 保存结果
    if args.output_file:
        print(f"\n步骤5: 保存预测结果")
        
        # 生成输出文件名
        model_suffix = "base" if args.base_model_only else "finetuned"
        prompt_suffix = "few_shot" if args.use_few_shot else "zero_shot"
        
        output_path = Path(args.output_file)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 修改文件名加入配置信息
        new_stem = f"{output_path.stem}_{model_suffix}_{prompt_suffix}"
        output_path = output_dir / f"{new_stem}.jsonl"
        
        # 保存预测详情
        print(f"  保存到: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in predictions:
                f.write(json.dumps(pred, ensure_ascii=False) + '\n')    
        
        # 保存评估摘要
        summary_path = output_dir / f"{new_stem}_summary.json"
        print(f"  保存摘要到: {summary_path}")
        with open(summary_path, 'w', encoding='utf-8') as f:
            # 构建summary - 直接使用results中计算好的指标
            summary = {
                'config': {
                    'model_type': model_suffix,
                    'prompt_type': prompt_suffix,
                    'temperature': args.temperature,
                    'num_samples': len(predictions)
                },
                'overall': results['overall'],
                'by_candidate': results['by_candidate'],
                'by_strategy': results['by_strategy'],
                'by_year': results['by_year']
            }
            
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"  预测结果已保存: {output_path}")
        print(f"  评估摘要已保存: {summary_path}")
    
    print("\n" + "="*80)
    print("✅ 推理完成！")
    print("="*80)


if __name__ == "__main__":
    main()

