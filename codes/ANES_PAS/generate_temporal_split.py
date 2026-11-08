#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
ANES时间序列划分 - 训练集和测试集生成

按时间划分数据集，确保：
1. 训练集只包含历史年份
2. 测试集包含未来年份
3. 避免数据泄漏
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Tuple
from generate_finetune_data import ANESFinetuneDataGenerator


class TemporalSplitGenerator:
    """时间序列划分生成器"""
    
    def __init__(self, data_path: str):
        """初始化"""
        self.df = pd.read_csv(data_path, low_memory=False)
        
        # 数据清洗（同generate_finetune_data.py）
        if 'support_score' in self.df.columns:
            self.df['support_score'] = pd.to_numeric(self.df['support_score'], errors='coerce')
            original_len = len(self.df)
            self.df = self.df.dropna(subset=['support_score'])
            if len(self.df) < original_len:
                print(f"⚠️  移除了 {original_len - len(self.df)} 条support_score缺失的记录")
        
        # 确保year是数值类型
        self.df['year'] = pd.to_numeric(self.df['year'], errors='coerce')
        
        print(f"加载数据: {len(self.df)} 条记录")
        
        # 分析年份分布
        self._analyze_years()
    
    def _analyze_years(self):
        """分析数据中的年份分布"""
        print("\n" + "=" * 80)
        print("数据年份分布")
        print("=" * 80)
        
        year_counts = self.df['year'].value_counts().sort_index()
        
        for year, count in year_counts.items():
            print(f"  {int(year)}: {count:6d} 条记录")
        
        print(f"\n  年份范围: {int(year_counts.index.min())} - {int(year_counts.index.max())}")
        print(f"  总年份数: {len(year_counts)}")
        
        # 分析候选人分布
        print("\n候选人分布（前10）:")
        candidate_counts = self.df['candidate'].value_counts().head(10)
        for candidate, count in candidate_counts.items():
            print(f"  {candidate:20s}: {count:6d}")
    
    def create_temporal_split(self, 
                             train_years: List[int] = None,
                             test_years: List[int] = None,
                             train_ratio: float = 0.8) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        创建时间序列划分
        
        Args:
            train_years: 训练集年份列表，None则自动划分
            test_years: 测试集年份列表，None则自动划分
            train_ratio: 自动划分时训练集比例
        
        Returns:
            (train_df, test_df)
        """
        if train_years is not None and test_years is not None:
            # 手动指定年份
            train_df = self.df[self.df['year'].isin(train_years)].copy()
            test_df = self.df[self.df['year'].isin(test_years)].copy()
        else:
            # 自动按时间划分
            years = sorted(self.df['year'].unique())
            split_idx = int(len(years) * train_ratio)
            
            train_years = years[:split_idx]
            test_years = years[split_idx:]
            
            train_df = self.df[self.df['year'].isin(train_years)].copy()
            test_df = self.df[self.df['year'].isin(test_years)].copy()
        
        print("\n" + "=" * 80)
        print("时间序列划分结果")
        print("=" * 80)
        print(f"\n训练集年份: {sorted([int(y) for y in train_years])}")
        print(f"训练集大小: {len(train_df)} 条记录")
        
        print(f"\n测试集年份: {sorted([int(y) for y in test_years])}")
        print(f"测试集大小: {len(test_df)} 条记录")
        
        print(f"\n划分比例: {len(train_df)/(len(train_df)+len(test_df))*100:.1f}% / "
              f"{len(test_df)/(len(train_df)+len(test_df))*100:.1f}%")
        
        return train_df, test_df
    
    def generate_finetune_dataset(self, 
                                 df: pd.DataFrame,
                                 output_path: str,
                                 strategies: List[str] = None,
                                 max_samples: int = None):
        """
        为给定的数据框生成微调数据集
        
        Args:
            df: 数据框
            output_path: 输出路径
            strategies: 采样策略
            max_samples: 最大样本数
        """
        # 保存为临时CSV
        temp_path = '/tmp/temp_anes_data.csv'
        df.to_csv(temp_path, index=False)
        
        # 使用ANESFinetuneDataGenerator生成
        generator = ANESFinetuneDataGenerator(temp_path)
        dataset = generator.generate_full_dataset(output_path, strategies=strategies)
        
        # 如果需要限制样本数
        if max_samples and len(dataset) > max_samples:
            print(f"\n🎲 随机下采样到 {max_samples} 个样本...")
            dataset = list(np.random.choice(dataset, max_samples, replace=False))
            
            # 重新保存
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in dataset:
                    f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            print(f"✅ 已保存下采样数据集")
        
        return dataset
    
    def generate_test_dataset(self,
                             test_df: pd.DataFrame,
                             output_path: str,
                             include_all: bool = False):
        """
        生成测试数据集（不同于训练集生成）
        
        测试集特点：
        1. 包含所有测试样本（不采样）或均衡采样
        2. 保留完整metadata用于评估
        3. 按候选人、年份分组
        
        Args:
            test_df: 测试数据框
            output_path: 输出路径
            include_all: 是否包含所有样本（默认False，会按候选人均衡采样）
        """
        print("\n" + "=" * 80)
        print("生成测试数据集")
        print("=" * 80)
        
        # 保存临时文件
        temp_path = '/tmp/temp_test_data.csv'
        test_df.to_csv(temp_path, index=False)
        
        # 创建生成器
        generator = ANESFinetuneDataGenerator(temp_path)
        
        samples = []
        
        if include_all:
            # 包含所有测试样本
            print(f"生成所有 {len(test_df)} 个测试样本...")
            
            for idx, row in test_df.iterrows():
                try:
                    prompt = generator._create_prompt_from_row(row, include_features='core')
                    response = generator._create_response(row['support_score'])
                    
                    samples.append({
                        'instruction': prompt,
                        'input': '',
                        'output': response,
                        'metadata': {
                            'strategy': 'test_all',
                            'year': int(row['year']),
                            'candidate': row.get('candidate', 'unknown'),
                            'true_score': float(row['support_score']),
                            'row_id': idx
                        }
                    })
                except Exception as e:
                    continue
        else:
            # 按候选人均衡采样
            print("按候选人均衡采样测试集...")
            
            candidates = test_df['candidate'].value_counts()
            print(f"\n候选人数量: {len(candidates)}")
            
            # 每个候选人采样相同数量
            samples_per_candidate = min(100, test_df.groupby('candidate').size().min())
            print(f"每个候选人采样: {samples_per_candidate} 个样本")
            
            for candidate in candidates.index:
                cand_df = test_df[test_df['candidate'] == candidate]
                cand_samples = cand_df.sample(min(samples_per_candidate, len(cand_df)))
                
                for idx, row in cand_samples.iterrows():
                    try:
                        prompt = generator._create_prompt_from_row(row, include_features='core')
                        response = generator._create_response(row['support_score'])
                        
                        samples.append({
                            'instruction': prompt,
                            'input': '',
                            'output': response,
                            'metadata': {
                                'strategy': 'test_balanced',
                                'year': int(row['year']),
                                'candidate': candidate,
                                'true_score': float(row['support_score']),
                                'row_id': idx
                            }
                        })
                    except Exception as e:
                        continue
                
                print(f"  {candidate:20s}: {len([s for s in samples if s['metadata']['candidate'] == candidate])} 样本")
        
        # 保存
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 测试集已保存: {output_path}")
        print(f"总样本数: {len(samples)}")
        
        return samples


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ANES时间序列划分数据生成')
    parser.add_argument('--data', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/processed/anes_all_years_with_features.csv',
                       help='输入数据路径')
    parser.add_argument('--output-dir', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/finetune',
                       help='输出目录')
    
    # 时间划分选项
    parser.add_argument('--train-years', type=int, nargs='+',
                       default=None,
                       help='训练集年份（例如：2016 2018 2020）')
    parser.add_argument('--test-years', type=int, nargs='+',
                       default=None,
                       help='测试集年份（例如：2022）')
    parser.add_argument('--auto-split', action='store_true',
                       help='自动按时间划分（默认80/20）')
    
    # 数据生成选项
    parser.add_argument('--train-strategies', type=str, nargs='+',
                       default=None,
                       help='训练集采样策略')
    parser.add_argument('--train-samples', type=int, default=None,
                       help='训练集最大样本数')
    parser.add_argument('--test-all', action='store_true',
                       help='测试集包含所有样本（默认按候选人均衡采样）')
    
    parser.add_argument('--preset', type=str, 
                       choices=['predict_2022', 'predict_2024', 'cross_validation'],
                       help='使用预设配置')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ANES时间序列划分 - 训练/测试集生成器")
    print("=" * 80)
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器
    generator = TemporalSplitGenerator(args.data)
    
    # 根据预设或参数确定划分方式
    if args.preset == 'predict_2022':
        print("\n📋 使用预设: 预测2022年（训练：≤2021）")
        train_years = [2016, 2018, 2019, 2020, 2021]
        test_years = [2022]
    elif args.preset == 'predict_2024':
        print("\n📋 使用预设: 预测2024年（训练：≤2022）")
        # 注意：需要确认数据中是否有2024年数据
        available_years = sorted(generator.df['year'].unique())
        if 2024 in available_years:
            train_years = [y for y in available_years if y < 2024]
            test_years = [2024]
        else:
            print("⚠️  数据中没有2024年，改用2022作为测试集")
            train_years = [2016, 2018, 2019, 2020, 2021]
            test_years = [2022]
    elif args.preset == 'cross_validation':
        print("\n📋 使用预设: 交叉验证（多个测试年份）")
        # 每年轮流作为测试集
        all_years = sorted(generator.df['year'].unique())
        print(f"可用年份: {[int(y) for y in all_years]}")
        print("请在交叉验证模式下多次运行，每次指定不同的test_years")
        return
    else:
        # 使用命令行参数
        train_years = args.train_years
        test_years = args.test_years
    
    # 创建划分
    train_df, test_df = generator.create_temporal_split(
        train_years=train_years,
        test_years=test_years
    )
    
    # 生成训练集
    print("\n" + "=" * 80)
    print("生成训练集")
    print("=" * 80)
    
    train_output = output_dir / 'train_dataset.jsonl'
    train_dataset = generator.generate_finetune_dataset(
        df=train_df,
        output_path=str(train_output),
        strategies=args.train_strategies,
        max_samples=args.train_samples
    )
    
    # 生成测试集
    test_output = output_dir / 'test_dataset.jsonl'
    test_dataset = generator.generate_test_dataset(
        test_df=test_df,
        output_path=str(test_output),
        include_all=args.test_all
    )
    
    # 如果train_years或test_years为None，从实际数据中提取
    if train_years is None or test_years is None:
        actual_train_years = sorted(train_df['year'].unique())
        actual_test_years = sorted(test_df['year'].unique())
    else:
        actual_train_years = sorted(train_years)
        actual_test_years = sorted(test_years)
    
    # 生成元数据
    metadata = {
        'train_years': [int(y) for y in actual_train_years],
        'test_years': [int(y) for y in actual_test_years],
        'train_size': len(train_dataset),
        'test_size': len(test_dataset),
        'train_file': str(train_output),
        'test_file': str(test_output),
        'strategies': args.train_strategies or 'default',
    }
    
    metadata_path = output_dir / 'split_metadata.json'
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("✅ 完成！")
    print("=" * 80)
    print(f"\n📁 输出文件:")
    print(f"  训练集: {train_output}")
    print(f"  测试集: {test_output}")
    print(f"  元数据: {metadata_path}")
    
    print(f"\n📊 数据统计:")
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    print(f"\n🎯 后续步骤:")
    print(f"  1. 用训练集微调模型")
    print(f"  2. 用测试集评估模型")
    print(f"  3. 分析预测误差")


if __name__ == '__main__':
    main()

