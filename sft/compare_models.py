#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
模型性能对比脚本

对比不同模型在ANES任务上的表现：
- Llama3.1 Base (Zero-shot / Few-shot)
- Llama3.1 Fine-tuned (Zero-shot / Few-shot)
- Qwen3 (Zero-shot / Few-shot)
"""

import json
import pandas as pd
from pathlib import Path
import argparse


def load_summary(file_path: str) -> dict:
    """加载评估汇总JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_metrics(summary: dict, model_name: str) -> dict:
    """从汇总中提取关键指标"""
    eval_data = summary.get('evaluation', {})
    overall = eval_data.get('overall', {})
    
    return {
        'Model': model_name,
        'Samples': overall.get('count', 0),
        'MAE': round(overall.get('mae', 0), 2),
        'RMSE': round(overall.get('rmse', 0), 2),
        'Accuracy': f"{overall.get('accuracy', 0)*100:.2f}%"
    }


def create_comparison_table(results_dict: dict) -> pd.DataFrame:
    """创建对比表格"""
    rows = []
    for model_name, file_path in results_dict.items():
        try:
            summary = load_summary(file_path)
            metrics = extract_metrics(summary, model_name)
            rows.append(metrics)
        except Exception as e:
            print(f"警告：无法加载 {model_name}: {e}")
    
    df = pd.DataFrame(rows)
    return df


def create_detailed_comparison(results_dict: dict) -> dict:
    """创建详细对比数据"""
    detailed = {}
    
    for model_name, file_path in results_dict.items():
        try:
            summary = load_summary(file_path)
            eval_data = summary.get('evaluation', {})
            
            detailed[model_name] = {
                'overall': eval_data.get('overall', {}),
                'by_candidate': eval_data.get('by_candidate', {}),
                'by_strategy': eval_data.get('by_strategy', {}),
                'by_year': eval_data.get('by_year', {})
            }
        except Exception as e:
            print(f"警告：无法加载 {model_name}: {e}")
    
    return detailed


def print_comparison_table(df: pd.DataFrame):
    """打印格式化的对比表格"""
    print("\n" + "="*80)
    print("模型性能对比 - 整体指标")
    print("="*80)
    print()
    print(df.to_string(index=False))
    print()


def print_detailed_comparison(detailed: dict):
    """打印详细对比信息"""
    if not detailed:
        return
    
    # 按候选人对比
    print("\n" + "="*80)
    print("按候选人分层对比")
    print("="*80)
    
    # 收集所有候选人
    all_candidates = set()
    for model_data in detailed.values():
        all_candidates.update(model_data['by_candidate'].keys())
    
    for candidate in sorted(all_candidates):
        print(f"\n{candidate}:")
        rows = []
        for model_name, data in detailed.items():
            if candidate in data['by_candidate']:
                metrics = data['by_candidate'][candidate]
                rows.append({
                    'Model': model_name,
                    'Count': metrics['count'],
                    'MAE': f"{metrics['mae']:.2f}",
                    'RMSE': f"{metrics['rmse']:.2f}",
                    'Acc': f"{metrics['accuracy']*100:.2f}%"
                })
        
        if rows:
            df = pd.DataFrame(rows)
            print(df.to_string(index=False))


def save_comparison_csv(df: pd.DataFrame, detailed: dict, output_file: str):
    """保存对比结果为CSV"""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 保存整体对比
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n对比表格已保存: {output_path}")
    
    # 保存按候选人的详细对比
    if detailed:
        all_candidates = set()
        for model_data in detailed.values():
            all_candidates.update(model_data['by_candidate'].keys())
        
        candidate_rows = []
        for candidate in sorted(all_candidates):
            for model_name, data in detailed.items():
                if candidate in data['by_candidate']:
                    metrics = data['by_candidate'][candidate]
                    candidate_rows.append({
                        'Candidate': candidate,
                        'Model': model_name,
                        'Count': metrics['count'],
                        'MAE': round(metrics['mae'], 2),
                        'RMSE': round(metrics['rmse'], 2),
                        'Accuracy': f"{metrics['accuracy']*100:.2f}%"
                    })
        
        if candidate_rows:
            candidate_df = pd.DataFrame(candidate_rows)
            candidate_file = output_path.parent / f"{output_path.stem}_by_candidate.csv"
            candidate_df.to_csv(candidate_file, index=False, encoding='utf-8-sig')
            print(f"候选人分层对比已保存: {candidate_file}")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='对比不同模型的性能')
    
    parser.add_argument('--llama_base_zero', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/enhanced_test_dataset_predictions_base_zero_shot_summary.json',
                       help='Llama3 Base Zero-shot结果')
    parser.add_argument('--llama_base_few', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/enhanced_test_dataset_predictions_base_few_shot_summary.json',
                       help='Llama3 Base Few-shot结果')
    parser.add_argument('--llama_ft_zero', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/enhanced_test_dataset_predictions_finetuned_zero_shot_summary.json',
                       help='Llama3 Fine-tuned Zero-shot结果')
    parser.add_argument('--llama_ft_few', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/anes_llama3_lora/enhanced_test_dataset_predictions_finetuned_few_shot_summary.json',
                       help='Llama3 Fine-tuned Few-shot结果')
    parser.add_argument('--qwen_zero', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/qwen3_results/qwen3_predictions_zero_shot_summary.json',
                       help='Qwen3 Zero-shot结果')
    parser.add_argument('--qwen_few', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/qwen3_results/qwen3_predictions_few_shot_summary.json',
                       help='Qwen3 Few-shot结果')
    parser.add_argument('--output', type=str,
                       default='/home/cyyuan/FAS_PRO/sft/sft_models/model_comparison.csv',
                       help='输出CSV文件路径')
    
    return parser.parse_args()


def main():
    """主函数"""
    args = parse_args()
    
    print("="*80)
    print("模型性能对比分析")
    print("="*80)
    
    # 定义要对比的模型及其结果文件
    results_dict = {
        'Llama3.1 Base (Zero-shot)': args.llama_base_zero,
        'Llama3.1 Base (Few-shot)': args.llama_base_few,
        'Llama3.1 Fine-tuned (Zero-shot)': args.llama_ft_zero,
        'Llama3.1 Fine-tuned (Few-shot)': args.llama_ft_few,
        'Qwen3 (Zero-shot)': args.qwen_zero,
        'Qwen3 (Few-shot)': args.qwen_few,
    }
    
    # 创建对比表格
    df = create_comparison_table(results_dict)
    
    # 创建详细对比
    detailed = create_detailed_comparison(results_dict)
    
    # 打印结果
    print_comparison_table(df)
    print_detailed_comparison(detailed)
    
    # 保存为CSV
    save_comparison_csv(df, detailed, args.output)
    
    print("\n" + "="*80)
    print("对比分析完成！")
    print("="*80)


if __name__ == '__main__':
    main()

