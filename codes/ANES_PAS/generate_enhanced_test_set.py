#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
生成增强版测试数据集
- 更多候选人（至少6个）
- 更大样本量
- 更全面的特征覆盖
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import List, Dict
from generate_finetune_data import ANESFinetuneDataGenerator


class EnhancedTestSetGenerator:
    """增强版测试集生成器"""
    
    def __init__(self, data_path: str):
        self.df = pd.read_csv(data_path, low_memory=False)
        
        # 数据清洗
        if 'support_score' in self.df.columns:
            self.df['support_score'] = pd.to_numeric(self.df['support_score'], errors='coerce')
            # 移除缺失值
            self.df = self.df.dropna(subset=['support_score'])
            # 移除异常值（support_score应该在0-100之间）
            original_len = len(self.df)
            self.df = self.df[(self.df['support_score'] >= 0) & (self.df['support_score'] <= 100)]
            if len(self.df) < original_len:
                print(f"⚠️  移除了 {original_len - len(self.df)} 条异常support_score记录 (<0 或 >100)")
        
        # 数值类型转换
        numeric_cols = ['age_resp', 'age_cand', 'year', 'pid7', 'ideo5', 
                       'party_match', 'ideology_distance_7']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"加载数据: {len(self.df)} 条记录")
        
        # 初始化prompt生成器
        self.prompt_generator = ANESFinetuneDataGenerator(data_path)
    
    def select_diverse_candidates(self, n_candidates: int = 8, 
                                  test_years: List[int] = None,
                                  min_samples: int = 50) -> List[str]:
        """选择多样化的候选人
        
        策略：
        1. 如果test_years为None，从所有年份选择
        2. 包含主要候选人和次要候选人
        3. 包含不同党派
        4. 确保每个候选人有足够样本
        """
        # 选择数据源
        if test_years:
            test_df = self.df[self.df['year'].isin(test_years)]
            print(f"\n从测试年份选择候选人: {test_years}")
        else:
            test_df = self.df
            print(f"\n从所有年份选择候选人")
        
        # 统计每个候选人的样本数
        candidate_counts = test_df['candidate'].value_counts()
        
        # 只保留样本数足够的候选人
        eligible_candidates = candidate_counts[candidate_counts >= min_samples]
        
        print(f"\n可用候选人 (样本数 >= {min_samples}):")
        for cand, count in eligible_candidates.head(20).items():
            print(f"  {cand:15s}: {count:5d} 样本")
        
        # 手动选择以确保多样性和代表性
        priority_candidates = [
            'trump', 'biden', 'obama', 'harris', 
            'sanders', 'warren', 'buttigieg', 'yang',
            'pence', 'rubio', 'pelosi', 'fauci',
            'dem', 'rep', 'hrc', 'cruz'
        ]
        
        selected = []
        for cand in priority_candidates:
            if cand in eligible_candidates.index and len(selected) < n_candidates:
                selected.append(cand)
        
        # 如果还不够，从剩余的中选择样本量最大的
        if len(selected) < n_candidates:
            remaining = [c for c in eligible_candidates.index if c not in selected]
            selected.extend(remaining[:n_candidates - len(selected)])
        
        print(f"\n✅ 选中 {len(selected)} 个候选人: {selected}")
        return selected[:n_candidates]
    
    def generate_stratified_samples(self, 
                                   candidates: List[str],
                                   n_per_candidate: int = 150,
                                   test_years: List[int] = None) -> List[Dict]:
        """为每个候选人生成分层采样的测试样本
        
        分层维度：
        1. 党派匹配 (party_match)
        2. 意识形态 (ideo5)
        3. 支持度区间 (score bins)
        4. 年龄段 (age groups)
        5. 教育水平 (education)
        6. 种族 (race)
        """
        print("\n" + "="*80)
        print("生成分层测试样本")
        print("="*80)
        
        all_samples = []
        if test_years:
            test_df = self.df[self.df['year'].isin(test_years)]
        else:
            test_df = self.df
        
        for candidate in candidates:
            cand_df = test_df[test_df['candidate'] == candidate].copy()
            
            if len(cand_df) == 0:
                print(f"⚠️  候选人 {candidate} 无数据，跳过")
                continue
            
            print(f"\n处理候选人: {candidate} (总样本: {len(cand_df)})")
            
            # 分层采样策略
            samples_collected = []
            
            # === 1. 党派分层 (最重要) ===
            party_strata = {
                'same_party': cand_df[cand_df['party_match'] == 1],
                'diff_party': cand_df[cand_df['party_match'] == 0],
                'independent': cand_df[cand_df['party_match'] == -1]
            }
            
            n_party = n_per_candidate // 3
            for stratum_name, stratum_df in party_strata.items():
                if len(stratum_df) > 0:
                    sampled = stratum_df.sample(min(n_party, len(stratum_df)))
                    samples_collected.append(sampled)
                    print(f"  - {stratum_name}: {len(sampled)} 样本")
            
            # === 2. 意识形态分层 ===
            if len(samples_collected) < n_per_candidate // 2:
                ideo_strata = []
                for ideo in [1, 2, 3, 4, 5]:  # Very Liberal -> Very Conservative
                    ideo_df = cand_df[cand_df['ideo5'] == ideo]
                    if len(ideo_df) > 0:
                        n_ideo = min(20, len(ideo_df))
                        sampled = ideo_df.sample(n_ideo)
                        samples_collected.append(sampled)
            
            # === 3. 支持度区间分层 ===
            score_bins = [(0, 20), (20, 40), (40, 60), (60, 80), (80, 100)]
            for low, high in score_bins:
                bin_df = cand_df[(cand_df['support_score'] >= low) & 
                                (cand_df['support_score'] < high)]
                if len(bin_df) > 0:
                    n_bin = min(15, len(bin_df))
                    sampled = bin_df.sample(n_bin)
                    samples_collected.append(sampled)
            
            # === 4. 年龄段分层 ===
            age_bins = [(18, 30), (30, 45), (45, 60), (60, 100)]
            for low, high in age_bins:
                age_df = cand_df[(cand_df['age_resp'] >= low) & 
                                (cand_df['age_resp'] < high)]
                if len(age_df) > 0:
                    n_age = min(15, len(age_df))
                    sampled = age_df.sample(n_age)
                    samples_collected.append(sampled)
            
            # === 5. 教育分层 ===
            if 'educ' in cand_df.columns:
                for educ_level in cand_df['educ'].dropna().unique():
                    educ_df = cand_df[cand_df['educ'] == educ_level]
                    if len(educ_df) > 0:
                        n_educ = min(10, len(educ_df))
                        sampled = educ_df.sample(n_educ)
                        samples_collected.append(sampled)
            
            # === 6. 种族分层 ===
            if 'race_resp' in cand_df.columns:
                for race in cand_df['race_resp'].dropna().unique():
                    race_df = cand_df[cand_df['race_resp'] == race]
                    if len(race_df) > 0:
                        n_race = min(10, len(race_df))
                        sampled = race_df.sample(n_race)
                        samples_collected.append(sampled)
            
            # 合并并去重
            combined_df = pd.concat(samples_collected).drop_duplicates()
            
            # 如果样本太多，随机下采样到目标数量
            if len(combined_df) > n_per_candidate:
                combined_df = combined_df.sample(n_per_candidate)
            
            print(f"  ✅ 最终样本数: {len(combined_df)}")
            
            # 转换为JSONL格式
            for idx, row in combined_df.iterrows():
                try:
                    # 过滤异常值
                    if row['support_score'] > 100:
                        continue
                    
                    # 使用core特征级别（包含关键交互特征）
                    prompt = self.prompt_generator._create_prompt_from_row(
                        row, include_features='core'
                    )
                    response = self.prompt_generator._create_response(row['support_score'])
                    
                    all_samples.append({
                        'instruction': prompt,
                        'input': '',
                        'output': response,
                        'metadata': {
                            'strategy': 'enhanced_test_stratified',
                            'candidate': candidate,
                            'year': int(row['year']),
                            'true_score': float(row['support_score']),
                            'party_match': int(row['party_match']) if pd.notna(row.get('party_match')) else None,
                            'ideology': int(row['ideo5']) if pd.notna(row.get('ideo5')) else None,
                            'age_group': self._get_age_group(row.get('age_resp')),
                            'education': int(row['educ']) if pd.notna(row.get('educ')) else None,
                            'race': int(row['race_resp']) if pd.notna(row.get('race_resp')) else None,
                            'row_id': int(idx)
                        }
                    })
                except Exception as e:
                    continue
        
        return all_samples
    
    def _get_age_group(self, age):
        """将年龄转换为年龄组"""
        if pd.isna(age):
            return None
        age = int(age)
        if age < 30:
            return '18-29'
        elif age < 45:
            return '30-44'
        elif age < 60:
            return '45-59'
        else:
            return '60+'
    
    def generate_edge_cases(self, candidates: List[str], 
                           n_per_candidate: int = 30) -> List[Dict]:
        """生成边界和极端案例"""
        print("\n生成边界和极端案例...")
        
        samples = []
        
        for candidate in candidates:
            cand_df = self.df[self.df['candidate'] == candidate]
            
            if len(cand_df) == 0:
                continue
            
            # 1. 支持度边界 (18-22, 38-42, 58-62, 78-82)
            boundary_df = cand_df[
                ((cand_df['support_score'] >= 18) & (cand_df['support_score'] <= 22)) |
                ((cand_df['support_score'] >= 38) & (cand_df['support_score'] <= 42)) |
                ((cand_df['support_score'] >= 58) & (cand_df['support_score'] <= 62)) |
                ((cand_df['support_score'] >= 78) & (cand_df['support_score'] <= 82))
            ]
            
            if len(boundary_df) > 0:
                sampled = boundary_df.sample(min(n_per_candidate // 2, len(boundary_df)))
                
                for idx, row in sampled.iterrows():
                    try:
                        prompt = self.prompt_generator._create_prompt_from_row(row)
                        response = self.prompt_generator._create_response(row['support_score'])
                        
                        samples.append({
                            'instruction': prompt,
                            'input': '',
                            'output': response,
                            'metadata': {
                                'strategy': 'edge_cases_boundary',
                                'candidate': candidate,
                                'year': int(row['year']),
                                'true_score': float(row['support_score']),
                                'row_id': int(idx)
                            }
                        })
                    except:
                        continue
            
            # 2. 极端案例 (<10 或 >90)
            extreme_df = cand_df[
                (cand_df['support_score'] <= 10) | 
                (cand_df['support_score'] >= 90)
            ]
            
            if len(extreme_df) > 0:
                sampled = extreme_df.sample(min(n_per_candidate // 2, len(extreme_df)))
                
                for idx, row in sampled.iterrows():
                    try:
                        prompt = self.prompt_generator._create_prompt_from_row(row)
                        response = self.prompt_generator._create_response(row['support_score'])
                        
                        samples.append({
                            'instruction': prompt,
                            'input': '',
                            'output': response,
                            'metadata': {
                                'strategy': 'edge_cases_extreme',
                                'candidate': candidate,
                                'year': int(row['year']),
                                'true_score': float(row['support_score']),
                                'row_id': int(idx)
                            }
                        })
                    except:
                        continue
        
        print(f"  生成 {len(samples)} 个边界/极端样本")
        return samples
    
    def analyze_coverage(self, samples: List[Dict]):
        """分析测试集覆盖度"""
        print("\n" + "="*80)
        print("测试集覆盖度分析")
        print("="*80)
        
        df = pd.DataFrame([s['metadata'] for s in samples])
        
        print(f"\n总样本数: {len(samples)}")
        
        print(f"\n候选人分布:")
        candidate_counts = df['candidate'].value_counts()
        for cand, count in candidate_counts.items():
            pct = count / len(df) * 100
            print(f"  {cand:15s}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n年份分布:")
        year_counts = df['year'].value_counts().sort_index()
        for year, count in year_counts.items():
            pct = count / len(df) * 100
            print(f"  {year}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n党派匹配分布:")
        if 'party_match' in df.columns:
            party_counts = df['party_match'].value_counts()
            party_labels = {1: '同党派', 0: '不同党派', -1: '独立选民'}
            for pm, count in party_counts.items():
                if pd.notna(pm):
                    pct = count / len(df) * 100
                    label = party_labels.get(int(pm), 'Unknown')
                    print(f"  {label}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n意识形态分布:")
        if 'ideology' in df.columns:
            ideo_counts = df['ideology'].value_counts().sort_index()
            ideo_labels = {1: 'Very Liberal', 2: 'Liberal', 3: 'Moderate', 
                          4: 'Conservative', 5: 'Very Conservative'}
            for ideo, count in ideo_counts.items():
                if pd.notna(ideo):
                    pct = count / len(df) * 100
                    label = ideo_labels.get(int(ideo), 'Unknown')
                    print(f"  {label}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n年龄组分布:")
        if 'age_group' in df.columns:
            age_counts = df['age_group'].value_counts()
            for age_group, count in sorted(age_counts.items()):
                if pd.notna(age_group):
                    pct = count / len(df) * 100
                    print(f"  {age_group}: {count:4d} ({pct:5.1f}%)")
        
        print(f"\n支持度分布:")
        scores = [s['metadata']['true_score'] for s in samples]
        print(f"  均值: {np.mean(scores):.2f}")
        print(f"  标准差: {np.std(scores):.2f}")
        print(f"  最小值: {np.min(scores):.2f}")
        print(f"  最大值: {np.max(scores):.2f}")
        
        # 支持度区间
        bins = [0, 20, 40, 60, 80, 100]
        labels = ['0-20', '20-40', '40-60', '60-80', '80-100']
        score_bins = pd.cut(scores, bins=bins, labels=labels, include_lowest=True)
        print(f"\n支持度区间分布:")
        for label in labels:
            count = (score_bins == label).sum()
            pct = count / len(scores) * 100
            print(f"  {label}: {count:4d} ({pct:5.1f}%)")
        
        print("="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='生成增强版测试数据集')
    parser.add_argument('--data', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/processed/anes_all_years_with_features.csv',
                       help='输入数据路径')
    parser.add_argument('--output', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/finetune/enhanced_test_dataset.jsonl',
                       help='输出路径')
    parser.add_argument('--n-candidates', type=int, default=8,
                       help='候选人数量（默认8）')
    parser.add_argument('--n-per-candidate', type=int, default=150,
                       help='每个候选人的样本数（默认150）')
    parser.add_argument('--test-years', type=int, nargs='+',
                       default=None,
                       help='测试年份（默认None表示所有年份）')
    parser.add_argument('--include-edges', action='store_true',
                       help='包含边界和极端案例')
    
    args = parser.parse_args()
    
    print("="*80)
    print("增强版测试数据集生成器")
    print("="*80)
    print(f"输入数据: {args.data}")
    print(f"输出路径: {args.output}")
    print(f"候选人数: {args.n_candidates}")
    print(f"每候选人样本数: {args.n_per_candidate}")
    print(f"测试年份: {args.test_years}")
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 初始化生成器
    generator = EnhancedTestSetGenerator(args.data)
    
    # 选择候选人
    candidates = generator.select_diverse_candidates(
        n_candidates=args.n_candidates,
        test_years=args.test_years
    )
    
    # 生成分层样本
    samples = generator.generate_stratified_samples(
        candidates=candidates,
        n_per_candidate=args.n_per_candidate,
        test_years=args.test_years
    )
    
    # 添加边界和极端案例
    if args.include_edges:
        edge_samples = generator.generate_edge_cases(
            candidates=candidates,
            n_per_candidate=30
        )
        samples.extend(edge_samples)
    
    # 去重
    print(f"\n去重前: {len(samples)} 样本")
    unique_samples = []
    seen_ids = set()
    for sample in samples:
        row_id = sample['metadata']['row_id']
        if row_id not in seen_ids:
            unique_samples.append(sample)
            seen_ids.add(row_id)
    
    print(f"去重后: {len(unique_samples)} 样本")
    
    # 打乱顺序
    np.random.shuffle(unique_samples)
    
    # 保存
    with open(args.output, 'w', encoding='utf-8') as f:
        for sample in unique_samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"\n✅ 测试数据集已保存: {args.output}")
    
    # 分析覆盖度
    generator.analyze_coverage(unique_samples)
    
    # 显示示例
    print("\n" + "="*80)
    print("样本示例")
    print("="*80)
    for i, sample in enumerate(unique_samples[:2]):
        print(f"\n【样本 {i+1}】")
        print(f"候选人: {sample['metadata']['candidate']}")
        print(f"年份: {sample['metadata']['year']}")
        print(f"真实分数: {sample['metadata']['true_score']}")
        print(f"\n{sample['instruction']}")
        print(f"\n答案: {sample['output']}")
        print("-"*80)
    
    print("\n✅ 全部完成！")


if __name__ == '__main__':
    main()

