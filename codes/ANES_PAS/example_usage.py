#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES微调数据集使用示例

展示如何加载和使用生成的微调数据集
"""

import json
import pandas as pd
from pathlib import Path


def load_finetune_dataset(filepath: str):
    """加载JSONL格式的微调数据集"""
    samples = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            samples.append(json.loads(line))
    return samples


def analyze_dataset(samples):
    """分析数据集的基本统计信息"""
    print("=" * 80)
    print("数据集统计分析")
    print("=" * 80)
    
    # 总样本数
    print(f"\n总样本数: {len(samples)}")
    
    # 策略分布
    print("\n策略分布:")
    strategy_counts = {}
    for sample in samples:
        strategy = sample['metadata'].get('strategy', 'unknown')
        strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
    
    for strategy, count in sorted(strategy_counts.items(), key=lambda x: -x[1]):
        percentage = count / len(samples) * 100
        print(f"  {strategy:25s}: {count:5d} ({percentage:5.1f}%)")
    
    # 支持度分布
    print("\n支持度分布:")
    scores = [sample['metadata'].get('true_score', 0) for sample in samples]
    print(f"  平均值: {sum(scores)/len(scores):.1f}")
    print(f"  最小值: {min(scores):.1f}")
    print(f"  最大值: {max(scores):.1f}")
    
    # 按区间统计
    bins = [(0, 20, "很低"), (20, 40, "较低"), (40, 60, "中等"), 
            (60, 80, "较高"), (80, 100, "很高")]
    print("\n  按区间分布:")
    for low, high, label in bins:
        count = sum(1 for s in scores if low <= s < high)
        percentage = count / len(scores) * 100
        print(f"    {label} [{low:3d}-{high:3d}): {count:5d} ({percentage:5.1f}%)")


def convert_to_training_format(samples, format_type='alpaca'):
    """
    转换为不同训练框架的格式
    
    Args:
        samples: 原始样本列表
        format_type: 'alpaca', 'openai', 'llama2'
    """
    if format_type == 'alpaca':
        # Alpaca格式（instruction-input-output）
        return samples  # 已经是这个格式
    
    elif format_type == 'openai':
        # OpenAI微调格式
        converted = []
        for sample in samples:
            converted.append({
                "messages": [
                    {"role": "system", "content": "You are an expert in political science specializing in voter behavior prediction."},
                    {"role": "user", "content": sample['instruction']},
                    {"role": "assistant", "content": sample['output']}
                ]
            })
        return converted
    
    elif format_type == 'llama2':
        # Llama2对话格式
        converted = []
        for sample in samples:
            text = f"<s>[INST] {sample['instruction']} [/INST] {sample['output']} </s>"
            converted.append({"text": text})
        return converted
    
    else:
        raise ValueError(f"Unknown format: {format_type}")


def save_converted_dataset(samples, output_path: str, format_type='alpaca'):
    """保存转换后的数据集"""
    converted = convert_to_training_format(samples, format_type)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for item in converted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存{format_type}格式数据集: {output_path}")


def show_examples(samples, n=3):
    """显示数据集样本"""
    print("\n" + "=" * 80)
    print(f"数据集样本 (显示前{n}个)")
    print("=" * 80)
    
    for i, sample in enumerate(samples[:n], 1):
        print(f"\n【样本 {i}】")
        print(f"策略: {sample['metadata'].get('strategy', 'unknown')}")
        print(f"真实分数: {sample['metadata'].get('true_score', 'N/A')}")
        print(f"\n输入 (Instruction):")
        print(sample['instruction'])
        print(f"\n输出 (Output): {sample['output']}")
        print("-" * 80)


def main():
    """主函数"""
    # 数据路径
    data_path = '/home/cyyuan/FAS_PRO/data/anes/finetune/anes_finetune_dataset.jsonl'
    
    print("=" * 80)
    print("ANES微调数据集 - 使用示例")
    print("=" * 80)
    
    # 1. 加载数据集
    print(f"\n📁 加载数据集: {data_path}")
    samples = load_finetune_dataset(data_path)
    print(f"✅ 成功加载 {len(samples)} 个样本")
    
    # 2. 分析数据集
    analyze_dataset(samples)
    
    # 3. 显示示例
    show_examples(samples, n=2)
    
    # 4. 转换格式示例
    print("\n" + "=" * 80)
    print("格式转换示例")
    print("=" * 80)
    
    output_dir = Path(data_path).parent
    
    # 转换为OpenAI格式
    openai_path = output_dir / 'anes_finetune_openai.jsonl'
    save_converted_dataset(samples, str(openai_path), format_type='openai')
    
    # 转换为Llama2格式
    llama2_path = output_dir / 'anes_finetune_llama2.jsonl'
    save_converted_dataset(samples, str(llama2_path), format_type='llama2')
    
    print("\n✅ 完成!")
    
    # 5. 提示后续步骤
    print("\n" + "=" * 80)
    print("后续步骤建议")
    print("=" * 80)
    print("""
1. 微调模型:
   - 使用OpenAI格式: openai api fine_tunes.create -t anes_finetune_openai.jsonl -m gpt-3.5-turbo
   - 使用Alpaca格式: python train_alpaca.py --data_path anes_finetune_dataset.jsonl
   - 使用Llama2: python train_llama2.py --data_path anes_finetune_llama2.jsonl

2. 评估模型:
   - 保留部分测试集（如2022年数据）
   - 计算MAE (Mean Absolute Error)
   - 分析不同策略的表现

3. 迭代改进:
   - 根据模型表现，调整采样策略
   - 增加困难案例的比例
   - 尝试不同的prompt格式（minimal/core/full）
    """)


if __name__ == '__main__':
    main()

