#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
使用Qwen3-4B-Instruct模型进行ANES选民支持度预测

对比Qwen3基础模型与Llama3微调模型的性能
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
from collections import defaultdict


def custom_accuracy(true_scores, pred_scores):
    """
    自定义准确率计算
    
    判断规则：
    1. 预测值和真实值在同一个20分段内（0-20, 20-40, 40-60, 60-80, 80-100）
    2. 或者预测值与真实值的绝对误差 ≤ 10
    
    满足任一条件即认为预测正确
    """
    correct = 0
    total = len(true_scores)
    
    for true_val, pred_val in zip(true_scores, pred_scores):
        true_bin = min(int(true_val // 20), 4)
        pred_bin = min(int(pred_val // 20), 4)
        abs_error = abs(true_val - pred_val)
        
        if true_bin == pred_bin or abs_error <= 10:
            correct += 1
    
    return correct / total if total > 0 else 0.0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用Qwen3进行ANES模型推理')
    
    parser.add_argument('--model_name', type=str,
                       default='Qwen/Qwen3-4B-Instruct-2507',
                       help='Qwen3模型名称或路径')
    parser.add_argument('--test_file', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='测试数据文件')
    parser.add_argument('--output_file', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/dp_predictions.jsonl',
                       help='预测结果输出文件（JSONL格式）')
    parser.add_argument('--summary_file', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/dp_predictions_summary.json',
                       help='评估结果汇总文件（JSON格式）')
    parser.add_argument('--max_new_tokens', type=int,
                       default=512,
                       help='最大生成token数（DeepSeek-R1需要更多tokens输出推理过程）')
    parser.add_argument('--temperature', type=float,
                       default=0.3,
                       help='生成温度（越低越确定）')
    parser.add_argument('--num_samples', type=int,
                       default=None,
                       help='测试样本数（None表示全部）')
    parser.add_argument('--use_few_shot', action='store_true',
                       help='是否使用few-shot examples（默认zero-shot）')
    
    return parser.parse_args()


def load_qwen3_model(model_name: str):
    """加载Qwen3模型和tokenizer"""
    print(f"加载Qwen3模型: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("模型加载完成！")
    return model, tokenizer


def create_prompt(sample: dict, use_few_shot: bool = False) -> str:
    """创建推理prompt（与inference_anes.py保持一致）"""
    system_prompt = (
        "You are an expert in political science specializing in voter behavior prediction. "
        "Your task is to predict a voter's support score (0-100) for a political candidate "
        "based on the voter's demographics, political views, and the candidate's characteristics. "
        "\n\nIMPORTANT: Respond with ONLY a single number between 0 and 100. "
        "Do NOT include any reasoning, explanation, or additional text. "
        "Just output the number directly."
    )
    
    few_shot_prompt = ""
    if use_few_shot:
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
            "- Strong partisans show more extreme scores (0-20 or 80-100)\n"
            "- Lean partisans: scores typically 20-90 range\n"
            "- Independents: scores typically 30-70, centered around 40-50\n"
            "- Ideology distance matters within same party\n"
            "- Demographics have secondary but important effects\n\n"
        )
    
    # 获取instruction部分（包含选民和候选人信息）
    user_content = sample.get('instruction', '')
    
    # 组合完整prompt
    if use_few_shot:
        full_prompt = f"{system_prompt}\n\n{few_shot_prompt}\nNow predict:\n{user_content}"
    else:
        full_prompt = f"{system_prompt}\n\n{user_content}"
    
    return full_prompt


def extract_score(response: str) -> int:
    """从模型响应中提取分数"""
    import re
    
    # 去除空白字符
    response = response.strip()
    
    # DeepSeek-R1特殊处理：移除<think>标签内的内容
    # DeepSeek-R1可能输出 <think>推理过程</think> 实际答案
    if '<think>' in response.lower():
        # 提取</think>之后的内容
        think_pattern = r'<think>.*?</think>'
        response = re.sub(think_pattern, '', response, flags=re.IGNORECASE | re.DOTALL).strip()
    
    # 如果响应以推理性语言开头（如"Okay"、"Let me"等），尝试找最后一个数字
    reasoning_keywords = ['okay', 'let me', 'i need to', 'so i', 'first', 'the answer']
    is_reasoning = any(response.lower().startswith(kw) for kw in reasoning_keywords)
    
    # 尝试匹配所有0-100的数字
    numbers = re.findall(r'\b(\d+)\b', response)
    
    if numbers:
        # 如果看起来是推理输出，取最后一个数字（通常是最终答案）
        # 否则取第一个数字
        score_candidates = [int(n) for n in numbers if 0 <= int(n) <= 100]
        
        if score_candidates:
            score = score_candidates[-1] if is_reasoning else score_candidates[0]
            return max(0, min(100, score))
    
    # 如果没有找到数字，返回50作为默认值
    print(f"  警告：无法从响应中提取分数: {response[:100]}")
    return 50


def generate_prediction(model, tokenizer, sample: dict, args) -> int:
    """使用Qwen3模型生成预测"""
    
    # 创建prompt
    prompt = create_prompt(sample, use_few_shot=args.use_few_shot)
    
    # 使用Qwen3的chat template格式
    messages = [
        {"role": "user", "content": prompt}
    ]
    
    # 应用chat template
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)
    
    # 生成
    # 检测是否是DeepSeek-R1模型，如果是则关闭reasoning
    is_deepseek_r1 = "deepseek" in args.model_name.lower() and "r1" in args.model_name.lower()
    
    with torch.no_grad():
        if is_deepseek_r1:
            # DeepSeek-R1特殊处理：使用较低的temperature和repetition_penalty来减少推理输出
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=0.1,  # 降低temperature使输出更确定
                do_sample=False,  # 使用贪心解码
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.0,
                top_p=0.9
            )
        else:
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                do_sample=True if args.temperature > 0 else False,
                pad_token_id=tokenizer.eos_token_id
            )
    
    # 解码响应（只取新生成的部分）
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    )
    
    # 提取分数
    score = extract_score(response)
    
    return score


def evaluate_predictions(predictions: list) -> dict:
    """评估预测结果"""
    
    # 提取真实值和预测值
    true_scores = [p['true_score'] for p in predictions]
    pred_scores = [p['predicted_score'] for p in predictions]
    
    # 计算整体指标
    mae = mean_absolute_error(true_scores, pred_scores)
    rmse = np.sqrt(np.mean((np.array(true_scores) - np.array(pred_scores)) ** 2))
    accuracy = custom_accuracy(true_scores, pred_scores)
    
    # 按candidate分层统计
    by_candidate = defaultdict(lambda: {'true': [], 'pred': []})
    for p in predictions:
        candidate = p['metadata']['candidate']
        by_candidate[candidate]['true'].append(p['true_score'])
        by_candidate[candidate]['pred'].append(p['predicted_score'])
    
    candidate_metrics = {}
    for candidate, scores in by_candidate.items():
        candidate_metrics[candidate] = {
            'count': len(scores['true']),
            'mae': float(mean_absolute_error(scores['true'], scores['pred'])),
            'rmse': float(np.sqrt(np.mean((np.array(scores['true']) - np.array(scores['pred'])) ** 2))),
            'accuracy': float(custom_accuracy(scores['true'], scores['pred']))
        }
    
    # 按strategy分层统计
    by_strategy = defaultdict(lambda: {'true': [], 'pred': []})
    for p in predictions:
        strategy = p['metadata']['strategy']
        by_strategy[strategy]['true'].append(p['true_score'])
        by_strategy[strategy]['pred'].append(p['predicted_score'])
    
    strategy_metrics = {}
    for strategy, scores in by_strategy.items():
        strategy_metrics[strategy] = {
            'count': len(scores['true']),
            'mae': float(mean_absolute_error(scores['true'], scores['pred'])),
            'rmse': float(np.sqrt(np.mean((np.array(scores['true']) - np.array(scores['pred'])) ** 2))),
            'accuracy': float(custom_accuracy(scores['true'], scores['pred']))
        }
    
    # 按year分层统计
    by_year = defaultdict(lambda: {'true': [], 'pred': []})
    for p in predictions:
        year = p['metadata']['year']
        by_year[year]['true'].append(p['true_score'])
        by_year[year]['pred'].append(p['predicted_score'])
    
    year_metrics = {}
    for year, scores in by_year.items():
        year_metrics[str(year)] = {
            'count': len(scores['true']),
            'mae': float(mean_absolute_error(scores['true'], scores['pred'])),
            'rmse': float(np.sqrt(np.mean((np.array(scores['true']) - np.array(scores['pred'])) ** 2))),
            'accuracy': float(custom_accuracy(scores['true'], scores['pred']))
        }
    
    return {
        'overall': {
            'count': len(predictions),
            'mae': float(mae),
            'rmse': float(rmse),
            'accuracy': float(accuracy)
        },
        'by_candidate': candidate_metrics,
        'by_strategy': strategy_metrics,
        'by_year': year_metrics
    }


def main():
    """主函数"""
    args = parse_args()
    
    print("="*60)
    print("Qwen3模型ANES推理")
    print("="*60)
    
    # 1. 加载模型
    print("\n步骤1: 加载Qwen3模型...")
    model, tokenizer = load_qwen3_model(args.model_name)
    
    # 2. 加载测试数据
    print(f"\n步骤2: 加载测试数据: {args.test_file}")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"  加载了 {len(test_data)} 条测试样本")
    
    # 3. 进行推理
    print(f"\n步骤3: 开始推理...")
    print(f"  Few-shot模式: {'是' if args.use_few_shot else '否'}")
    print(f"  Temperature: {args.temperature}")
    
    predictions = []
    
    for sample in tqdm(test_data, desc="推理进度"):
        try:
            # 生成预测
            predicted_score = generate_prediction(model, tokenizer, sample, args)
            
            # 记录结果
            prediction = {
                'instruction': sample['instruction'],
                'true_score': float(sample['output']),
                'predicted_score': predicted_score,
                'metadata': sample['metadata']
            }
            predictions.append(prediction)
            
        except Exception as e:
            print(f"\n  错误：处理样本时出错: {e}")
            continue
    
    print(f"\n  完成 {len(predictions)}/{len(test_data)} 条预测")
    
    # 4. 评估结果
    print(f"\n步骤4: 计算评估指标...")
    evaluation = evaluate_predictions(predictions)
    
    # 5. 保存结果
    print(f"\n步骤5: 保存结果...")
    
    # 保存详细预测结果
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    print(f"  详细预测结果: {output_path}")
    
    # 保存评估汇总
    summary_path = Path(args.summary_file)
    summary_data = {
        'model': args.model_name,
        'test_file': args.test_file,
        'num_samples': len(predictions),
        'use_few_shot': args.use_few_shot,
        'temperature': args.temperature,
        'evaluation': evaluation
    }
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)
    print(f"  评估汇总: {summary_path}")
    
    # 6. 打印结果
    print("\n" + "="*60)
    print("评估结果")
    print("="*60)
    
    print(f"\n整体性能:")
    print(f"  样本数: {evaluation['overall']['count']}")
    print(f"  MAE: {evaluation['overall']['mae']:.2f}")
    print(f"  RMSE: {evaluation['overall']['rmse']:.2f}")
    print(f"  准确率: {evaluation['overall']['accuracy']:.2%}")
    
    print(f"\n按候选人分层:")
    for candidate, metrics in evaluation['by_candidate'].items():
        print(f"  {candidate}:")
        print(f"    样本数: {metrics['count']}, MAE: {metrics['mae']:.2f}, "
              f"RMSE: {metrics['rmse']:.2f}, 准确率: {metrics['accuracy']:.2%}")
    
    print(f"\n按策略分层:")
    for strategy, metrics in evaluation['by_strategy'].items():
        print(f"  {strategy}:")
        print(f"    样本数: {metrics['count']}, MAE: {metrics['mae']:.2f}, "
              f"RMSE: {metrics['rmse']:.2f}, 准确率: {metrics['accuracy']:.2%}")
    
    print(f"\n按年份分层:")
    for year, metrics in sorted(evaluation['by_year'].items()):
        print(f"  {year}:")
        print(f"    样本数: {metrics['count']}, MAE: {metrics['mae']:.2f}, "
              f"RMSE: {metrics['rmse']:.2f}, 准确率: {metrics['accuracy']:.2%}")
    
    print("\n" + "="*60)
    print("推理完成！")
    print("="*60)


if __name__ == '__main__':
    main()

