#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对比实验：微调模型 vs DeepSeek-R1-Distill-Llama-8B

测试更强大的基础模型（DeepSeek）与你的微调模型的效果对比

使用方法：
python compare_with_deepseek.py \
    --finetuned_base_model meta-llama/Llama-3.1-8B-Instruct \
    --finetuned_lora_path ./sft_models/anes_llama3_lora/final_model \
    --test_file /home/cyyuan/FAS_PRO/data/anes/finetune/test_dataset.jsonl
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import sys
import pandas as pd


def custom_accuracy(true_scores, pred_scores):
    """
    自定义准确率计算
    规则：在同一个20分段内 OR 误差≤10分 即认为正确
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
    parser = argparse.ArgumentParser(description='对比微调模型与DeepSeek模型')
    
    # 微调模型参数
    parser.add_argument('--finetuned_base_model', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='微调模型的基础模型')
    parser.add_argument('--finetuned_lora_path', type=str,
                       required=True,
                       help='微调模型的LoRA路径')
    
    # DeepSeek模型参数
    parser.add_argument('--deepseek_model', type=str,
                       default='deepseek-ai/DeepSeek-R1-Distill-Llama-8B',
                       help='DeepSeek模型路径')
    
    # 数据参数
    parser.add_argument('--test_file', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='测试数据文件')
    parser.add_argument('--output_dir', type=str,
                       default='/home/cyyuan/FAS_PRO/data/anes/finetune/deepseek_comparison',
                       help='输出目录')
    
    # 推理参数
    parser.add_argument('--max_new_tokens', type=int,
                       default=10,
                       help='最大生成token数')
    parser.add_argument('--temperature', type=float,
                       default=0.1,
                       help='生成温度')
    parser.add_argument('--num_samples', type=int,
                       default=None,
                       help='测试样本数（None表示全部）')
    
    return parser.parse_args()


def load_finetuned_model(base_model_path: str, lora_path: str):
    """加载微调模型"""
    print(f"\n{'='*80}")
    print("加载微调模型")
    print(f"{'='*80}")
    print(f"基础模型: {base_model_path}")
    print(f"LoRA适配器: {lora_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("  加载LoRA适配器...")
    model = PeftModel.from_pretrained(
        base_model,
        lora_path,
        torch_dtype=torch.bfloat16
    )
    
    print("  合并LoRA权重...")
    model = model.merge_and_unload()
    
    print("  ✅ 微调模型加载完成！")
    return model, tokenizer


def load_deepseek_model(model_path: str):
    """加载DeepSeek模型"""
    print(f"\n{'='*80}")
    print("加载DeepSeek模型")
    print(f"{'='*80}")
    print(f"模型: {model_path}")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=False,
        trust_remote_code=True
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("  加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    print("  ✅ DeepSeek模型加载完成！")
    return model, tokenizer


def create_prompt(sample: dict) -> str:
    """创建推理prompt"""
    system_prompt = (
        "You are an expert in political science specializing in voter behavior prediction. "
        "Your task is to predict a voter's support score (0-100) for a political candidate "
        "based on the voter's demographics, political views, and the candidate's characteristics. "
        "Respond with ONLY a number between 0 and 100."
    )
    
    background_prompt = """
    The data comes from American National Election Studies(ANES), 
    you need to consider many political and social factors, including the policies of the year and general approval ratings.
    """
    
    user_message = sample['instruction'] + sample['input'] + background_prompt
    
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
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
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")
        )
    
    generated = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    
    return response


def extract_score(response: str) -> float:
    """从模型响应中提取分数"""
    try:
        score = float(response.strip())
        score = max(0, min(100, score))
        return score
    except:
        import re
        numbers = re.findall(r'\d+\.?\d*', response)
        if numbers:
            score = float(numbers[0])
            return max(0, min(100, score))
        return 50.0


def evaluate_and_compare(predictions: list, test_data: list):
    """评估和对比两个模型"""
    finetuned_scores = []
    deepseek_scores = []
    true_scores = []
    
    for pred, sample in zip(predictions, test_data):
        finetuned_scores.append(pred['finetuned_score'])
        deepseek_scores.append(pred['deepseek_score'])
        true_scores.append(sample['metadata']['true_score'])
    
    # 计算指标
    results = {
        'finetuned': {
            'mae': mean_absolute_error(true_scores, finetuned_scores),
            'rmse': np.sqrt(np.mean((np.array(finetuned_scores) - np.array(true_scores))**2)),
            'accuracy': custom_accuracy(true_scores, finetuned_scores)
        },
        'deepseek': {
            'mae': mean_absolute_error(true_scores, deepseek_scores),
            'rmse': np.sqrt(np.mean((np.array(deepseek_scores) - np.array(true_scores))**2)),
            'accuracy': custom_accuracy(true_scores, deepseek_scores)
        }
    }
    
    # 计算提升
    results['improvement'] = {
        'mae': results['deepseek']['mae'] - results['finetuned']['mae'],
        'rmse': results['deepseek']['rmse'] - results['finetuned']['rmse'],
        'accuracy': results['finetuned']['accuracy'] - results['deepseek']['accuracy']
    }
    
    # 按候选人分层
    candidate_results = {}
    for pred, sample in zip(predictions, test_data):
        candidate = sample['metadata'].get('candidate', 'unknown')
        if candidate not in candidate_results:
            candidate_results[candidate] = {
                'true': [], 
                'finetuned': [], 
                'deepseek': []
            }
        candidate_results[candidate]['true'].append(sample['metadata']['true_score'])
        candidate_results[candidate]['finetuned'].append(pred['finetuned_score'])
        candidate_results[candidate]['deepseek'].append(pred['deepseek_score'])
    
    results['candidate_results'] = candidate_results
    
    # 按策略分层
    strategy_results = {}
    for pred, sample in zip(predictions, test_data):
        strategy = sample['metadata'].get('strategy', 'unknown')
        if strategy not in strategy_results:
            strategy_results[strategy] = {
                'true': [], 
                'finetuned': [], 
                'deepseek': []
            }
        strategy_results[strategy]['true'].append(sample['metadata']['true_score'])
        strategy_results[strategy]['finetuned'].append(pred['finetuned_score'])
        strategy_results[strategy]['deepseek'].append(pred['deepseek_score'])
    
    results['strategy_results'] = strategy_results
    
    return results


def print_results(results: dict):
    """打印对比结果"""
    print("\n" + "="*80)
    print("对比结果")
    print("="*80)
    
    # 整体指标
    print(f"\n{'指标':<25} {'DeepSeek':>15} {'微调模型':>15} {'微调优势':>15}")
    print("-" * 75)
    print(f"{'MAE':.<25} {results['deepseek']['mae']:>15.2f} {results['finetuned']['mae']:>15.2f} {results['improvement']['mae']:>15.2f}")
    print(f"{'RMSE':.<25} {results['deepseek']['rmse']:>15.2f} {results['finetuned']['rmse']:>15.2f} {results['improvement']['rmse']:>15.2f}")
    print(f"{'准确率(20分段/误差≤10)':.<25} {results['deepseek']['accuracy']:>14.1%} {results['finetuned']['accuracy']:>14.1%} {results['improvement']['accuracy']:>14.1%}")
    
    # 按候选人对比
    print(f"\n按候选人分层对比:")
    print(f"{'候选人':<15} {'DeepSeek MAE':>15} {'微调MAE':>15} {'微调优势':>12} {'样本数':>10}")
    print("-" * 70)
    for candidate, scores in sorted(results['candidate_results'].items()):
        deepseek_mae = mean_absolute_error(scores['true'], scores['deepseek'])
        finetuned_mae = mean_absolute_error(scores['true'], scores['finetuned'])
        improvement = deepseek_mae - finetuned_mae
        print(f"{candidate:<15} {deepseek_mae:>15.2f} {finetuned_mae:>15.2f} {improvement:>12.2f} {len(scores['true']):>10}")
    
    # 按策略对比
    print(f"\n按策略分层对比:")
    print(f"{'策略':<25} {'DeepSeek MAE':>15} {'微调MAE':>15} {'微调优势':>12} {'样本数':>10}")
    print("-" * 75)
    for strategy, scores in sorted(results['strategy_results'].items()):
        deepseek_mae = mean_absolute_error(scores['true'], scores['deepseek'])
        finetuned_mae = mean_absolute_error(scores['true'], scores['finetuned'])
        improvement = deepseek_mae - finetuned_mae
        print(f"{strategy:<25} {deepseek_mae:>15.2f} {finetuned_mae:>15.2f} {improvement:>12.2f} {len(scores['true']):>10}")


def save_results(results: dict, predictions: list, output_dir: Path):
    """保存结果"""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. 保存预测详情
    predictions_file = output_dir / 'predictions.jsonl'
    with open(predictions_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + '\n')
    
    # 2. 保存评估摘要
    summary = {
        'finetuned_model': {
            'mae': results['finetuned']['mae'],
            'rmse': results['finetuned']['rmse'],
            'accuracy': results['finetuned']['accuracy']
        },
        'deepseek_model': {
            'mae': results['deepseek']['mae'],
            'rmse': results['deepseek']['rmse'],
            'accuracy': results['deepseek']['accuracy']
        },
        'finetuned_advantage': {
            'mae_improvement': results['improvement']['mae'],
            'rmse_improvement': results['improvement']['rmse'],
            'accuracy_improvement': results['improvement']['accuracy']
        },
        'candidate_comparison': {
            candidate: {
                'deepseek_mae': mean_absolute_error(v['true'], v['deepseek']),
                'finetuned_mae': mean_absolute_error(v['true'], v['finetuned']),
                'deepseek_accuracy': custom_accuracy(v['true'], v['deepseek']),
                'finetuned_accuracy': custom_accuracy(v['true'], v['finetuned']),
                'mae_improvement': mean_absolute_error(v['true'], v['deepseek']) - mean_absolute_error(v['true'], v['finetuned']),
                'accuracy_improvement': custom_accuracy(v['true'], v['finetuned']) - custom_accuracy(v['true'], v['deepseek']),
                'sample_count': len(v['true'])
            }
            for candidate, v in results['candidate_results'].items()
        },
        'strategy_comparison': {
            strategy: {
                'deepseek_mae': mean_absolute_error(v['true'], v['deepseek']),
                'finetuned_mae': mean_absolute_error(v['true'], v['finetuned']),
                'deepseek_accuracy': custom_accuracy(v['true'], v['deepseek']),
                'finetuned_accuracy': custom_accuracy(v['true'], v['finetuned']),
                'mae_improvement': mean_absolute_error(v['true'], v['deepseek']) - mean_absolute_error(v['true'], v['finetuned']),
                'accuracy_improvement': custom_accuracy(v['true'], v['finetuned']) - custom_accuracy(v['true'], v['deepseek']),
                'sample_count': len(v['true'])
            }
            for strategy, v in results['strategy_results'].items()
        }
    }
    
    summary_file = output_dir / 'comparison_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    # 3. 生成CSV表格
    generate_csv_reports(summary, output_dir)
    
    print(f"\n✅ 结果已保存到: {output_dir}")
    print(f"  - 预测详情: {predictions_file}")
    print(f"  - 评估摘要: {summary_file}")


def generate_csv_reports(summary: dict, output_dir: Path):
    """生成CSV对比报告"""
    
    # 1. 整体对比
    overall_data = []
    for model_name, model_key in [('DeepSeek', 'deepseek_model'), ('微调模型', 'finetuned_model')]:
        overall_data.append({
            '模型': model_name,
            'MAE': round(summary[model_key]['mae'], 2),
            'RMSE': round(summary[model_key]['rmse'], 2),
            '准确率(%)': round(summary[model_key]['accuracy'] * 100, 2)
        })
    
    # 添加优势行
    overall_data.append({
        '模型': '微调优势',
        'MAE': round(summary['finetuned_advantage']['mae_improvement'], 2),
        'RMSE': round(summary['finetuned_advantage']['rmse_improvement'], 2),
        '准确率(%)': round(summary['finetuned_advantage']['accuracy_improvement'] * 100, 2)
    })
    
    df_overall = pd.DataFrame(overall_data)
    df_overall.to_csv(output_dir / 'comparison_overall.csv', index=False, encoding='utf-8-sig')
    
    # 2. 按候选人对比
    candidate_data = []
    for candidate, metrics in summary['candidate_comparison'].items():
        candidate_data.append({
            '候选人': candidate,
            '样本数': metrics['sample_count'],
            'DeepSeek_MAE': round(metrics['deepseek_mae'], 2),
            '微调_MAE': round(metrics['finetuned_mae'], 2),
            'MAE提升': round(metrics['mae_improvement'], 2),
            'DeepSeek_准确率(%)': round(metrics['deepseek_accuracy'] * 100, 2),
            '微调_准确率(%)': round(metrics['finetuned_accuracy'] * 100, 2),
            '准确率提升(%)': round(metrics['accuracy_improvement'] * 100, 2)
        })
    
    df_candidate = pd.DataFrame(candidate_data)
    df_candidate = df_candidate.sort_values('MAE提升', ascending=False)
    df_candidate.to_csv(output_dir / 'comparison_by_candidate.csv', index=False, encoding='utf-8-sig')
    
    # 3. 按策略对比
    strategy_data = []
    for strategy, metrics in summary['strategy_comparison'].items():
        strategy_data.append({
            '策略': strategy,
            '样本数': metrics['sample_count'],
            'DeepSeek_MAE': round(metrics['deepseek_mae'], 2),
            '微调_MAE': round(metrics['finetuned_mae'], 2),
            'MAE提升': round(metrics['mae_improvement'], 2),
            'DeepSeek_准确率(%)': round(metrics['deepseek_accuracy'] * 100, 2),
            '微调_准确率(%)': round(metrics['finetuned_accuracy'] * 100, 2),
            '准确率提升(%)': round(metrics['accuracy_improvement'] * 100, 2)
        })
    
    df_strategy = pd.DataFrame(strategy_data)
    df_strategy = df_strategy.sort_values('MAE提升', ascending=False)
    df_strategy.to_csv(output_dir / 'comparison_by_strategy.csv', index=False, encoding='utf-8-sig')
    
    print(f"  - CSV报告: comparison_overall.csv, comparison_by_candidate.csv, comparison_by_strategy.csv")


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES模型对比实验：微调模型 vs DeepSeek-R1-Distill-Llama-8B")
    print("="*80)
    
    # 1. 加载微调模型
    finetuned_model, finetuned_tokenizer = load_finetuned_model(
        args.finetuned_base_model,
        args.finetuned_lora_path
    )
    
    # 2. 加载DeepSeek模型
    deepseek_model, deepseek_tokenizer = load_deepseek_model(args.deepseek_model)
    
    # 3. 加载测试数据
    print(f"\n{'='*80}")
    print("加载测试数据")
    print(f"{'='*80}")
    test_data = []
    with open(args.test_file, 'r', encoding='utf-8') as f:
        for line in f:
            test_data.append(json.loads(line))
    
    if args.num_samples:
        test_data = test_data[:args.num_samples]
    
    print(f"测试样本数: {len(test_data)}")
    
    # 4. 推理
    print(f"\n{'='*80}")
    print("开始推理")
    print(f"{'='*80}")
    predictions = []
    
    for i, sample in enumerate(tqdm(test_data, desc="预测进度")):
        prompt = create_prompt(sample)
        
        # 微调模型预测
        finetuned_response = predict(
            finetuned_model, 
            finetuned_tokenizer, 
            prompt, 
            args.max_new_tokens, 
            args.temperature
        )
        finetuned_score = extract_score(finetuned_response)
        
        # DeepSeek预测
        deepseek_response = predict(
            deepseek_model, 
            deepseek_tokenizer, 
            prompt, 
            args.max_new_tokens, 
            args.temperature
        )
        deepseek_score = extract_score(deepseek_response)
        
        predictions.append({
            'sample_id': i,
            'candidate': sample['metadata'].get('candidate', 'unknown'),
            'year': sample['metadata'].get('year', 'unknown'),
            'strategy': sample['metadata'].get('strategy', 'unknown'),
            'true_score': sample['metadata']['true_score'],
            'finetuned_score': finetuned_score,
            'deepseek_score': deepseek_score,
            'finetuned_response': finetuned_response,
            'deepseek_response': deepseek_response,
            'finetuned_error': abs(finetuned_score - sample['metadata']['true_score']),
            'deepseek_error': abs(deepseek_score - sample['metadata']['true_score'])
        })
    
    print(f"\n推理完成，共 {len(predictions)} 个样本")
    
    # 5. 评估和对比
    print(f"\n{'='*80}")
    print("评估和对比")
    print(f"{'='*80}")
    results = evaluate_and_compare(predictions, test_data)
    
    # 6. 打印结果
    print_results(results)
    
    # 7. 保存结果
    output_dir = Path(args.output_dir)
    save_results(results, predictions, output_dir)
    
    print("\n" + "="*80)
    print("✅ 对比实验完成！")
    print("="*80)


if __name__ == "__main__":
    main()

