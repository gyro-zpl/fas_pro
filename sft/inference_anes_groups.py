#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES群体态度预测 - 推理和评估脚本

功能：
1. 加载微调后的模型
2. 在测试集上进行推理
3. 计算详细的评估指标
4. 生成评估报告

使用方法：
python inference_anes_groups.py --model_path /path/to/model --test_file test.jsonl
"""

import torch
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from sklearn.metrics import mean_absolute_error, mean_squared_error


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ANES群体态度预测推理')
    
    parser.add_argument('--model_path', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/sft/sft_models/anes_groups_llama3_lora/final_model',
                       help='LoRA模型路径（adapter）')
    parser.add_argument('--base_model', type=str,
                       default='meta-llama/Llama-3.1-8B-Instruct',
                       help='基础模型路径')
    parser.add_argument('--test_file', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/extend_features/data/group_test_balanced_v3.jsonl',
                       help='测试数据文件')
    parser.add_argument('--output_dir', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/extend_features/results',
                       help='结果输出目录')
    parser.add_argument('--max_new_tokens', type=int,
                       default=10,
                       help='生成的最大token数')
    parser.add_argument('--batch_size', type=int,
                       default=4,
                       help='推理batch size')
    parser.add_argument('--use_few_shot', action='store_true',
                       help='是否使用few-shot examples（默认zero-shot）')
    parser.add_argument('--base_model_only', action='store_true',
                       help='仅使用base model，不加载LoRA（用于测试base model性能）')
    
    return parser.parse_args()


def load_model_and_tokenizer(lora_path, base_model_path, base_only=False):
    """
    加载模型和tokenizer
    
    Args:
        lora_path: LoRA adapter路径
        base_model_path: 基础模型路径
        base_only: 是否仅加载base model，不加载LoRA
    
    Returns:
        model, tokenizer
    """
    print(f"\n加载模型...")
    print(f"  基础模型: {base_model_path}")
    if not base_only:
        print(f"  LoRA路径: {lora_path}")
    else:
        print("  仅使用基础模型（不加载LoRA）")
    
    # 1. 加载tokenizer
    print("  加载tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_path,
        trust_remote_code=True
    )
    
    # 设置pad_token
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
    
    # 3. 如果不是base_only，加载并合并LoRA adapter
    if not base_only:
        print("  加载LoRA adapter...")
    model = PeftModel.from_pretrained(
            model,
        lora_path,
        torch_dtype=torch.bfloat16
    )
    
    # 4. 合并权重（提高推理速度）
    print("  合并LoRA权重到基础模型...")
    model = model.merge_and_unload()
    
    model.eval()
    print(f"  ✅ 模型加载完成")
    
    return model, tokenizer


def load_test_data(file_path):
    """加载测试数据"""
    print(f"\n加载测试数据: {file_path}")
    
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    print(f"  测试样本数: {len(data)}")
    return data


def generate_prediction(model, tokenizer, instruction, max_new_tokens=10, use_few_shot=False):
    """生成单个预测
    
    Args:
        model: 模型
        tokenizer: tokenizer
        instruction: 指令
        max_new_tokens: 最大生成token数
        use_few_shot: 是否使用few-shot examples
    """
    system_prompt = (
        "You are an expert in political science and social psychology specializing in group attitudes. "
        "Your task is to predict a voter's feeling thermometer score (0-100) toward a social group "
        "based on the voter's demographics, political views, and the group's characteristics. "
        "A score of 0 means very cold/unfavorable feelings, 100 means very warm/favorable feelings, "
        "and 50 means neutral. Respond with ONLY a number between 0 and 100."
    )
    
    few_shot_prompt = (
        "Here are 5 examples covering different score ranges:\n\n"
        
        "Example 1 - Very High (80-100):\n"
        "Voter: Strong Democrat, Liberal ideology, Age 35\n"
        "Group: Democratic Party\n"
        "Factors: Political alignment = Aligned, Strong partisan\n"
        "Predicted score: 95\n"
        "Reasoning: Strong partisan alignment with own party.\n\n"
        
        "Example 2 - Moderate-High (60-79):\n"
        "Voter: Lean Republican, Moderate ideology, Age 50\n"
        "Group: White Americans\n"
        "Factors: Racial in-group, Moderate political views\n"
        "Predicted score: 70\n"
        "Reasoning: In-group racial identity with moderate political stance.\n\n"
        
        "Example 3 - Neutral (40-59):\n"
        "Voter: Independent, Moderate ideology, Age 45\n"
        "Group: Asian Americans\n"
        "Factors: Independent voter, No strong political tie\n"
        "Predicted score: 50\n"
        "Reasoning: Independent voters tend toward neutral feelings on most groups.\n\n"
        
        "Example 4 - Moderate-Low (20-39):\n"
        "Voter: Republican, Conservative ideology, Age 55\n"
        "Group: Socialists\n"
        "Factors: Opposing ideology, Different political views\n"
        "Predicted score: 30\n"
        "Reasoning: Ideological opposition leads to cool feelings.\n\n"
        
        "Example 5 - Very Low (0-19):\n"
        "Voter: Strong Republican, Very Conservative, Age 60\n"
        "Group: Feminists\n"
        "Factors: Strong ideological opposition\n"
        "Predicted score: 10\n"
        "Reasoning: Strong conservative opposition to feminist ideology.\n\n"
        
        "Key patterns:\n"
        "- Score ranges: Very High (80-100), Moderate-High (60-79), Neutral (40-59), Moderate-Low (20-39), Very Low (0-19)\n"
        "- Political alignment is the strongest predictor\n"
        "- Strong partisans show more extreme scores than lean/independent\n"
        "- Racial/ethnic groups: generally neutral to positive (50-70) unless strong political factors\n"
        "- Ideological groups: strongly polarized by voter's own ideology\n\n"
    )
    
    # 根据use_few_shot参数决定是否包含few-shot examples
    if use_few_shot:
        system_content = f"{system_prompt}\n\n{few_shot_prompt}"
    else:
        system_content = system_prompt
    
    # 构建prompt
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_content}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{instruction}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.convert_tokens_to_ids("<|eot_id|>")  # Llama 3.1 的特殊 EOS token
        )
    
    # 解码输出（只取新生成的部分）
    generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    
    # 解析数字
    try:
        score = float(response.strip())
        score = max(0, min(100, score))
    except:
        score = 50.0  # 默认值
    
    return score, response


def custom_accuracy(true_scores, pred_scores):
    """
    自定义准确率：同一20分档 OR 误差≤10
    """
    correct = 0
    for true_val, pred_val in zip(true_scores, pred_scores):
        true_bin = min(int(true_val // 20), 4)
        pred_bin = min(int(pred_val // 20), 4)
        abs_error = abs(true_val - pred_val)
        
        if true_bin == pred_bin or abs_error <= 10:
            correct += 1
    
    return correct / len(true_scores) if len(true_scores) > 0 else 0.0


def evaluate_predictions(test_data, predictions, base_predictions=None):
    """
    评估预测结果（支持微调模型和base model对比）
    
    Args:
        test_data: 测试数据
        predictions: 微调模型预测列表
        base_predictions: base model预测列表（可选）
    """
    true_scores = [d['metadata']['true_score'] for d in test_data]
    
    # 微调模型评估
    ft_mae = mean_absolute_error(true_scores, predictions)
    ft_rmse = np.sqrt(mean_squared_error(true_scores, predictions))
    ft_acc = custom_accuracy(true_scores, predictions)
    
    # 检查base_predictions是否有效（不是None且不包含None值）
    has_valid_base = (base_predictions is not None and 
                      len(base_predictions) > 0 and 
                      base_predictions[0] is not None)
    
    results = {
        'finetuned': {
            'mae': ft_mae,
            'rmse': ft_rmse,
            'accuracy': ft_acc,
            'n_samples': len(true_scores)
        }
    }
    
    # 如果有base model预测，添加对比
    if has_valid_base:
        base_mae = mean_absolute_error(true_scores, base_predictions)
        base_rmse = np.sqrt(mean_squared_error(true_scores, base_predictions))
        base_acc = custom_accuracy(true_scores, base_predictions)
        
        results['base'] = {
            'mae': base_mae,
            'rmse': base_rmse,
            'accuracy': base_acc
        }
        results['improvement'] = {
            'mae': base_mae - ft_mae,
            'rmse': base_rmse - ft_rmse,
            'accuracy': ft_acc - base_acc
        }
    
    # 按群体分层
    group_metrics = {}
    for sample, true_score, ft_pred in zip(test_data, true_scores, predictions):
        group = sample['metadata']['group']
        if group not in group_metrics:
            group_metrics[group] = {'true': [], 'ft_pred': [], 'base_pred': []}
        group_metrics[group]['true'].append(true_score)
        group_metrics[group]['ft_pred'].append(ft_pred)
        
        # 如果有base预测，也添加进去（检查是否有效）
        if has_valid_base:
            idx = test_data.index(sample)
            group_metrics[group]['base_pred'].append(base_predictions[idx])
    
    # 计算每个群体的详细指标
    results['by_group'] = {}
    for group, metrics in group_metrics.items():
        ft_mae = mean_absolute_error(metrics['true'], metrics['ft_pred'])
        ft_rmse = np.sqrt(mean_squared_error(metrics['true'], metrics['ft_pred']))
        ft_acc = custom_accuracy(metrics['true'], metrics['ft_pred'])
        
        group_result = {
            'finetuned': {
                'mae': ft_mae,
                'rmse': ft_rmse,
                'accuracy': ft_acc
            },
            'n_samples': len(metrics['true'])
        }
        
        if has_valid_base and len(metrics['base_pred']) > 0:
            base_mae = mean_absolute_error(metrics['true'], metrics['base_pred'])
            base_rmse = np.sqrt(mean_squared_error(metrics['true'], metrics['base_pred']))
            base_acc = custom_accuracy(metrics['true'], metrics['base_pred'])
            
            group_result['base'] = {
                'mae': base_mae,
                'rmse': base_rmse,
                'accuracy': base_acc
            }
            group_result['improvement'] = {
                'mae': base_mae - ft_mae,
                'rmse': base_rmse - ft_rmse,
                'accuracy': ft_acc - base_acc
            }
        
        results['by_group'][group] = group_result
    
    # 按策略分层
    strategy_metrics = {}
    for sample, true_score, ft_pred in zip(test_data, true_scores, predictions):
        strategy = sample['metadata'].get('strategy', 'unknown')
        if strategy not in strategy_metrics:
            strategy_metrics[strategy] = {'true': [], 'ft_pred': [], 'base_pred': []}
        strategy_metrics[strategy]['true'].append(true_score)
        strategy_metrics[strategy]['ft_pred'].append(ft_pred)
        
        # 如果有base预测，也添加进去
        if has_valid_base:
            idx = test_data.index(sample)
            strategy_metrics[strategy]['base_pred'].append(base_predictions[idx])
    
    results['by_strategy'] = {}
    for strategy, metrics in strategy_metrics.items():
        ft_mae = mean_absolute_error(metrics['true'], metrics['ft_pred'])
        ft_rmse = np.sqrt(mean_squared_error(metrics['true'], metrics['ft_pred']))
        ft_acc = custom_accuracy(metrics['true'], metrics['ft_pred'])
        
        strategy_result = {
            'finetuned': {
                'mae': ft_mae,
                'rmse': ft_rmse,
                'accuracy': ft_acc
            },
            'n_samples': len(metrics['true'])
        }
        
        # 如果有base model预测，也计算base的指标
        if has_valid_base and len(metrics['base_pred']) > 0:
            base_mae = mean_absolute_error(metrics['true'], metrics['base_pred'])
            base_rmse = np.sqrt(mean_squared_error(metrics['true'], metrics['base_pred']))
            base_acc = custom_accuracy(metrics['true'], metrics['base_pred'])
            strategy_result['base'] = {
                'mae': base_mae,
                'rmse': base_rmse,
                'accuracy': base_acc
            }
            strategy_result['improvement'] = {
                'mae': base_mae - ft_mae,
                'rmse': base_rmse - ft_rmse,
                'accuracy': ft_acc - base_acc
            }
        
        results['by_strategy'][strategy] = strategy_result
    
    return results


def print_results(results):
    """打印评估结果（支持base model对比）"""
    print("\n" + "="*80)
    print("评估结果")
    print("="*80)
    
    # 整体对比
    if 'base' in results:
        print(f"\n{'指标':<20} {'Base Model':>15} {'Fine-tuned':>15} {'提升':>15}")
        print("-" * 70)
        print(f"{'MAE':.<20} {results['base']['mae']:>15.2f} {results['finetuned']['mae']:>15.2f} {results['improvement']['mae']:>15.2f}")
        print(f"{'RMSE':.<20} {results['base']['rmse']:>15.2f} {results['finetuned']['rmse']:>15.2f} {results['improvement']['rmse']:>15.2f}")
        print(f"{'准确率':.<20} {results['base']['accuracy']:>14.1%} {results['finetuned']['accuracy']:>14.1%} {results['improvement']['accuracy']:>14.1%}")
        print(f"{'样本数':.<20} {results['finetuned']['n_samples']:>15}")
    else:
        print(f"\n整体指标:")
        print(f"  MAE:              {results['finetuned']['mae']:.2f}")
        print(f"  RMSE:             {results['finetuned']['rmse']:.2f}")
        print(f"  准确率:            {results['finetuned']['accuracy']:.2%}")
        print(f"  样本数:           {results['finetuned']['n_samples']}")
    
    # 按群体分层对比
    print(f"\n按群体分层对比:")
    if 'base' in results:
        print(f"{'群体':<30} {'指标':<8} {'Base':>10} {'Fine-tuned':>12} {'提升':>10} {'样本数':>8}")
        print("-" * 85)
        for group in sorted(results['by_group'].keys()):
            metrics = results['by_group'][group]
            n = metrics['n_samples']
            
            # MAE行
            print(f"{group:<30} {'MAE':<8} {metrics['base']['mae']:>10.2f} {metrics['finetuned']['mae']:>12.2f} {metrics['improvement']['mae']:>10.2f} {n:>8}")
            # RMSE行
            print(f"{'':30} {'RMSE':<8} {metrics['base']['rmse']:>10.2f} {metrics['finetuned']['rmse']:>12.2f} {metrics['improvement']['rmse']:>10.2f}")
            # Accuracy行
            print(f"{'':30} {'Acc':<8} {metrics['base']['accuracy']:>9.1%} {metrics['finetuned']['accuracy']:>11.1%} {metrics['improvement']['accuracy']:>9.1%}")
            print("-" * 85)
    else:
        print(f"{'群体':<30} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'样本数':>8}")
        print("-" * 70)
        for group, metrics in sorted(results['by_group'].items()):
            print(f"{group:<30} {metrics['finetuned']['mae']:>10.2f} {metrics['finetuned']['rmse']:>10.2f} {metrics['finetuned']['accuracy']:>9.1%} {metrics['n_samples']:>8}")
    
    # 按策略分层对比
    print(f"\n按策略分层对比:")
    if 'base' in results:
        print(f"{'策略':<30} {'指标':<8} {'Base':>10} {'Fine-tuned':>12} {'提升':>10} {'样本数':>8}")
        print("-" * 85)
        for strategy in sorted(results['by_strategy'].keys()):
            metrics = results['by_strategy'][strategy]
            n = metrics['n_samples']
            
            # 检查是否有base model对比
            if 'base' in metrics:
                # MAE行
                print(f"{strategy:<30} {'MAE':<8} {metrics['base']['mae']:>10.2f} {metrics['finetuned']['mae']:>12.2f} {metrics['improvement']['mae']:>10.2f} {n:>8}")
                # RMSE行
                print(f"{'':30} {'RMSE':<8} {metrics['base']['rmse']:>10.2f} {metrics['finetuned']['rmse']:>12.2f} {metrics['improvement']['rmse']:>10.2f}")
                # Accuracy行
                print(f"{'':30} {'Acc':<8} {metrics['base']['accuracy']:>9.1%} {metrics['finetuned']['accuracy']:>11.1%} {metrics['improvement']['accuracy']:>9.1%}")
                print("-" * 85)
            else:
                # 只有fine-tuned
                print(f"{strategy:<30} {'MAE':<8} {'':>10} {metrics['finetuned']['mae']:>12.2f} {'':>10} {n:>8}")
                print(f"{'':30} {'RMSE':<8} {'':>10} {metrics['finetuned']['rmse']:>12.2f}")
                print(f"{'':30} {'Acc':<8} {'':>10} {metrics['finetuned']['accuracy']:>11.1%}")
                print("-" * 85)
    else:
        print(f"{'策略':<30} {'MAE':>10} {'RMSE':>10} {'Acc':>10} {'样本数':>8}")
        print("-" * 70)
    for strategy, metrics in sorted(results['by_strategy'].items()):
            print(f"{strategy:<30} {metrics['finetuned']['mae']:>10.2f} {metrics['finetuned']['rmse']:>10.2f} {metrics['finetuned']['accuracy']:>9.1%} {metrics['n_samples']:>8}")
    
    print("="*80)


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("ANES群体态度预测 - 推理评估")
    print("="*80)
    print(f"基础模型: {args.base_model}")
    print(f"LoRA路径: {args.model_path}")
    print(f"测试文件: {args.test_file}")
    print(f"输出目录: {args.output_dir}")
    print(f"Few-shot: {'是' if args.use_few_shot else '否'}")
    print(f"仅Base Model: {'是' if args.base_model_only else '否（使用Fine-tuned）'}")
    
    # 创建输出目录
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. 加载模型
    print("\n步骤1: 加载模型")
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.base_model, base_only=args.base_model_only)
    
    # 2. 如果不是base_model_only，加载base model用于对比
    base_model = None
    if not args.base_model_only:
        print("\n步骤2: 加载base model（用于对比）")
        print(f"  基础模型: {args.base_model}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    base_model.eval()
    print(f"  ✅ Base model加载完成")
    
    # 3. 加载测试数据
    test_data = load_test_data(args.test_file)
    
    # 4. 推理
    has_base_comparison = not args.base_model_only and base_model is not None
    if has_base_comparison:
        print(f"\n步骤3: 开始推理（Fine-tuned + Base model对比）")
    else:
        model_type = "Base Model" if args.base_model_only else "Fine-tuned Model"
        print(f"\n步骤3: 开始推理（{model_type}）")
    
    main_predictions = []
    base_predictions = [] if has_base_comparison else None
    raw_outputs = []
    
    for sample in tqdm(test_data, desc="推理进度"):
        instruction = sample['instruction'] + sample['input']
        
        # 主模型预测（带few-shot参数）
        main_score, raw_output = generate_prediction(
            model, tokenizer, instruction, args.max_new_tokens, use_few_shot=args.use_few_shot
        )
        main_predictions.append(main_score)
        raw_outputs.append(raw_output)
        
        # Base model预测（如果需要对比）
        if has_base_comparison:
            base_score, _ = generate_prediction(
                base_model, tokenizer, instruction, args.max_new_tokens, use_few_shot=args.use_few_shot
            )
            base_predictions.append(base_score)
        #break
    # 5. 评估
    print(f"\n步骤4: 计算评估指标...")
    if has_base_comparison:
        results = evaluate_predictions(test_data, main_predictions, base_predictions)
    else:
        # 如果没有base model对比，传None
        results = evaluate_predictions(test_data, main_predictions, None)
    
    # 6. 打印结果
    print_results(results)
    
    # 7. 保存结果
    # 生成文件名
    model_suffix = "base" if args.base_model_only else "finetuned"
    prompt_suffix = "few_shot" if args.use_few_shot else "zero_shot"
    output_file = Path(args.output_dir) / f"group_predictions_{model_suffix}_{prompt_suffix}.jsonl"
    
    print(f"\n步骤5: 保存预测结果到: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for idx, (sample, main_pred, raw) in enumerate(zip(test_data, main_predictions, raw_outputs)):
            result = {
                'instruction': sample['instruction'],
                'true_score': sample['metadata']['true_score'],
                'predicted_score': main_pred,
                'raw_output': raw,
                'group': sample['metadata']['group'],
                'strategy': sample['metadata'].get('strategy', 'unknown'),
                'error': abs(sample['metadata']['true_score'] - main_pred)
            }
            if has_base_comparison and base_predictions is not None:
                base_pred = base_predictions[idx]
                result['base_pred'] = base_pred
                result['base_error'] = abs(sample['metadata']['true_score'] - base_pred)
            f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    # 8. 保存汇总结果
    summary_file = Path(args.output_dir) / f"group_evaluation_summary_{model_suffix}_{prompt_suffix}.json"
    print(f"保存汇总结果到: {summary_file}")
    
    # 添加配置信息到results
    results['config'] = {
        'model_type': model_suffix,
        'prompt_type': prompt_suffix,
        'base_model': args.base_model,
        'lora_path': args.model_path if not args.base_model_only else None
    }
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "="*80)
    print("✅ 评估完成！")
    print("="*80)
    print(f"\n生成的文件:")
    print(f"  - 详细预测: {output_file}")
    print(f"  - 评估汇总: {summary_file}")
    
    if 'improvement' in results:
        print(f"\n关键提升:")
        print(f"  MAE提升:   {results['improvement']['mae']:+.2f}")
        print(f"  RMSE提升:  {results['improvement']['rmse']:+.2f}")
        print(f"  准确率提升: {results['improvement']['accuracy']:+.1%}")


if __name__ == "__main__":
    main()

