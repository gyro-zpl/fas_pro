"""
为ANES预测任务生成高质量微调数据集

策略：
1. 难样本挖掘：选择模型预测困难的样本
2. 边界样本增强：支持度在边界区域的样本
3. 多样化采样：确保覆盖各种特征组合
4. 错误样本纠正：模型预测错误的样本
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import json
from pathlib import Path


class ANESFinetuneDataGenerator:
    """ANES微调数据生成器"""
    
    def __init__(self, data_path: str):
        """
        初始化
        
        Args:
            data_path: ANES数据CSV路径
        """
        # 读取数据
        self.df = pd.read_csv(data_path, low_memory=False)
        
        # 确保support_score是数值类型
        if 'support_score' in self.df.columns:
            self.df['support_score'] = pd.to_numeric(self.df['support_score'], errors='coerce')
            # 删除support_score为NaN的行
            original_len = len(self.df)
            self.df = self.df.dropna(subset=['support_score'])
            if len(self.df) < original_len:
                print(f"⚠️  移除了 {original_len - len(self.df)} 条support_score缺失的记录")
        
        # 确保其他关键数值列也是正确类型
        numeric_cols = ['age_resp', 'age_cand', 'age_difference', 'party_match', 
                       'ideology_distance_7', 'ideology_same_side_7',
                       'demographic_similarity', 'is_incumbent', 'year']
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        print(f"加载数据: {len(self.df)} 条记录")
        
        # 定义支持度区间
        self.score_bins = [
            (0, 20, "cold"),       # 非常不支持
            (20, 40, "somewhat_cold"),  # 有点不支持
            (40, 60, "neutral"),   # 中立
            (60, 80, "somewhat_warm"),  # 有点支持
            (80, 100, "warm")      # 非常支持
        ]
    
    def _create_prompt_from_row(self, row: pd.Series, include_features: str = 'core') -> str:
        """
        从数据行创建prompt
        
        Args:
            row: 数据行
            include_features: 'minimal', 'core', 'full' - 包含的特征级别
        """
        
        # 辅助函数：安全转换为int
        def safe_int(val, default=0):
            try:
                if pd.isna(val):
                    return default
                return int(val)
            except (ValueError, TypeError):
                return default
        
        def safe_float(val, default=0.0):
            try:
                if pd.isna(val):
                    return default
                return float(val)
            except (ValueError, TypeError):
                return default
        
        # 基础信息
        age_resp = safe_int(row.get('age_resp'), 45)
        gender_resp = 'Male' if safe_int(row.get('gender_resp'), 1) == 1 else 'Female'
        race_map = {1: 'White', 2: 'Black', 3: 'Hispanic', 4: 'Asian', 5: 'Other'}
        race_resp = race_map.get(safe_int(row.get('race_resp'), 1), 'Unknown')
        
        # 教育和收入
        educ_map = {1: 'Elementary or less', 2: 'Some high school', 3: 'High school diploma', 
                   4: 'Some college', 5: "Bachelor's degree", 6: 'Graduate degree'}
        educ = educ_map.get(safe_int(row.get('educ'), 3), 'High school')
        income = safe_int(row.get('faminc'), 14)
        
        # 政治倾向
        pid7 = safe_int(row.get('pid7'), 4)
        party_labels = {1: 'Strong Democrat', 2: 'Democrat', 3: 'Lean Democrat', 
                       4: 'Independent', 5: 'Lean Republican', 6: 'Republican', 
                       7: 'Strong Republican'}
        party_affiliation = party_labels.get(pid7, 'Independent')
        
        ideo5 = safe_int(row.get('ideo5'), 3)
        ideology_labels = {1: 'Very Liberal', 2: 'Liberal', 3: 'Moderate', 
                          4: 'Conservative', 5: 'Very Conservative'}
        ideology = ideology_labels.get(ideo5, 'Moderate')
        
        # 候选人信息
        candidate = row.get('candidate', 'Unknown')
        year = safe_int(row.get('year'), 2020)
        age_cand = safe_int(row.get('age_cand'), 60)
        gender_cand = 'Male' if safe_int(row.get('gender_cand'), 1) == 1 else 'Female'
        race_cand = race_map.get(safe_int(row.get('race_cand'), 1), 'White')
        party_cand = 'Democrat' if safe_int(row.get('party'), 1) == 1 else 'Republican'
        
        # 位置信息
        position_map = {1: 'President', 2: 'Vice President', 3: 'Candidate',
                       4: 'Former President', 5: 'Speaker', 6: 'Senator',
                       7: 'Governor', 8: 'Justice', 9: 'Doctor', 10: 'Advisor'}
        position = position_map.get(safe_int(row.get('position_encoded'), 3), 'Candidate')
        
        # === 构建prompt（根据级别） ===
        
        if include_features == 'minimal':
            # 最小特征集：只有关键政治信息
            prompt = f"""Predict the support score (0-100) for this voter-candidate pair.

Voter: {age_resp}-year-old {gender_resp} {race_resp}, {party_affiliation}, {ideology}
Candidate: {candidate} ({party_cand} {position}, {age_cand} years old)
Year: {year}

Score (0-100):"""
        
        elif include_features == 'core':
            # 核心特征集：加入关键交互特征
            party_match = safe_int(row.get('party_match'), 0)
            party_match_text = "Same party" if party_match == 1 else \
                             "Independent voter" if party_match == -1 else "Different party"
            
            ideology_dist = safe_float(row.get('ideology_distance_7'), 0)
            age_diff = safe_int(row.get('age_difference'), 0)
            
            prompt = f"""Predict the support score (0-100) for this voter-candidate pair.

Voter Profile:
- Demographics: {age_resp}-year-old {gender_resp} {race_resp}
- Education: {educ}
- Political: {party_affiliation}, {ideology}

Candidate: {candidate}
- Party: {party_cand}
- Position: {position}  
- Age: {age_cand}

Key Interactions:
- Party alignment: {party_match_text}
- Ideology distance: {ideology_dist:.1f} (on 7-point scale)
- Age difference: {age_diff} years
- Year: {year}

Support score (0-100):"""
        
        else:  # 'full'
            # 完整特征集：包含所有关键信息
            party_match = safe_int(row.get('party_match'), 0)
            ideology_dist = safe_float(row.get('ideology_distance_7'), 0)
            age_diff = safe_int(row.get('age_difference'), 0)
            race_match = safe_int(row.get('race_match'), 0)
            gender_match = safe_int(row.get('gender_match'), 0)
            demographic_sim = safe_float(row.get('demographic_similarity'), 0.5)
            is_incumbent = safe_int(row.get('is_incumbent'), 0)
            
            prompt = f"""Predict the support score (0-100) for this voter-candidate pair in {year}.

VOTER PROFILE:
- Demographics: {age_resp}yo {gender_resp}, {race_resp}, {educ}
- Income level: {income}/28
- Political identity: {party_affiliation} ({pid7}/7)
- Ideology: {ideology} ({ideo5}/5)

CANDIDATE: {candidate}
- Party: {party_cand}
- Position: {position} {"(Incumbent)" if is_incumbent else ""}
- Demographics: {age_cand}yo {gender_cand}, {race_cand}

KEY MATCHING FACTORS:
- Party alignment: {"✓ Same" if party_match==1 else "✗ Different" if party_match==0 else "~ Independent"}
- Ideology distance: {ideology_dist:.1f}/7
- Age difference: {age_diff} years ({"same generation" if age_diff<16 else "adjacent gen" if age_diff<31 else "cross-gen"})
- Race match: {"✓" if race_match else "✗"}
- Gender match: {"✓" if gender_match else "✗"}
- Overall similarity: {demographic_sim:.2f}

Predict support score (0-100):"""
        
        return prompt
    
    def _create_response(self, score: float) -> str:
        """创建标准化的回复"""
        return str(int(score))
    
    def strategy_1_hard_samples(self, model_predictions: pd.DataFrame = None, 
                                top_k: int = 500) -> List[Dict]:
        """
        策略1: 难样本挖掘
        
        选择预测误差最大的样本（如果有预测结果）
        否则选择支持度在边界区域的样本
        """
        print("\n策略1: 难样本挖掘")
        
        samples = []
        
        if model_predictions is not None:
            # 如果有模型预测结果，选择误差最大的样本
            self.df['prediction_error'] = abs(
                self.df['support_score'] - model_predictions['predicted_score']
            )
            hard_samples = self.df.nlargest(top_k, 'prediction_error')
        else:
            # 否则选择边界区域样本（如45-55分，这些最难预测）
            boundary_mask = (
                ((self.df['support_score'] >= 18) & (self.df['support_score'] <= 22)) |
                ((self.df['support_score'] >= 38) & (self.df['support_score'] <= 42)) |
                ((self.df['support_score'] >= 58) & (self.df['support_score'] <= 62)) |
                ((self.df['support_score'] >= 78) & (self.df['support_score'] <= 82))
            )
            hard_samples = self.df[boundary_mask].sample(min(top_k, boundary_mask.sum()))
        
        for idx, row in hard_samples.iterrows():
            prompt = self._create_prompt_from_row(row)
            response = self._create_response(row['support_score'])
            
            samples.append({
                'instruction': prompt,
                'input': '',
                'output': response,
                'metadata': {
                    'strategy': 'hard_samples',
                    'true_score': float(row['support_score']),
                    'candidate': row['candidate'],
                    'year': int(row['year'])
                }
            })
        
        print(f"生成 {len(samples)} 个难样本")
        return samples
    
    def strategy_2_balanced_sampling(self, n_per_bin: int = 200) -> List[Dict]:
        """
        策略2: 平衡采样
        
        从每个支持度区间均匀采样
        """
        print("\n策略2: 平衡采样")
        
        samples = []
        
        for low, high, label in self.score_bins:
            mask = (self.df['support_score'] >= low) & (self.df['support_score'] < high)
            bin_samples = self.df[mask].sample(min(n_per_bin, mask.sum()))
            
            for idx, row in bin_samples.iterrows():
                prompt = self._create_prompt_from_row(row)
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'balanced_sampling',
                        'score_range': label,
                        'true_score': float(row['support_score'])
                    }
                })
            
            print(f"  {label} [{low}-{high}): {len(bin_samples)} 样本")
        
        print(f"总计生成 {len(samples)} 个平衡样本")
        return samples
    
    def strategy_3_diverse_features(self, n_samples: int = 500) -> List[Dict]:
        """
        策略3: 特征多样化采样
        
        确保覆盖各种特征组合
        """
        print("\n策略3: 特征多样化采样")
        
        samples = []
        
        # 定义要多样化的关键特征
        key_features = ['pid7', 'ideo5', 'educ', 'age_resp', 'race_resp']
        
        # 按关键特征分组
        for feature in key_features:
            if feature in self.df.columns:
                unique_values = self.df[feature].dropna().unique()
                n_per_value = max(1, n_samples // len(unique_values))
                
                for value in unique_values:
                    feature_samples = self.df[self.df[feature] == value].sample(
                        min(n_per_value, len(self.df[self.df[feature] == value]))
                    )
                    
                    for idx, row in feature_samples.iterrows():
                        if len(samples) >= n_samples:
                            break
                        
                        prompt = self._create_prompt_from_row(row)
                        response = self._create_response(row['support_score'])
                        
                        samples.append({
                            'instruction': prompt,
                            'input': '',
                            'output': response,
                            'metadata': {
                                'strategy': 'diverse_features',
                                'key_feature': feature,
                                'feature_value': str(value)
                            }
                        })
                    
                    if len(samples) >= n_samples:
                        break
        
        print(f"生成 {len(samples)} 个多样化样本")
        return samples
    
    def strategy_4_extreme_cases(self, n_samples: int = 200) -> List[Dict]:
        """
        策略4: 极端案例
        
        选择非常支持（>90）和非常反对（<10）的样本
        """
        print("\n策略4: 极端案例")
        
        samples = []
        
        # 极端支持
        extreme_support = self.df[self.df['support_score'] >= 90].sample(
            min(n_samples // 2, len(self.df[self.df['support_score'] >= 90]))
        )
        
        # 极端反对
        extreme_oppose = self.df[self.df['support_score'] <= 10].sample(
            min(n_samples // 2, len(self.df[self.df['support_score'] <= 10]))
        )
        
        for df_subset in [extreme_support, extreme_oppose]:
            for idx, row in df_subset.iterrows():
                prompt = self._create_prompt_from_row(row)
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'extreme_cases',
                        'true_score': float(row['support_score'])
                    }
                })
        
        print(f"生成 {len(samples)} 个极端案例样本")
        return samples
    
    def strategy_5_party_ideology_focused(self, n_samples: int = 600) -> List[Dict]:
        """
        策略5: 党派×意识形态组合采样
        
        根据特征文档，party_match和ideology_distance是最重要的特征
        重点采样各种党派-意识形态组合
        """
        print("\n策略5: 党派×意识形态组合采样")
        
        samples = []
        
        # 定义关键组合
        combinations = [
            # (party_match, ideology_same_side, 说明)
            (1, 1, 'perfect_align'),      # 完美对齐：同党派同意识形态
            (1, 0, 'same_party_diff_ideo'),  # 同党派但意识形态不同
            (0, 1, 'diff_party_same_ideo'),  # 不同党派但意识形态相同
            (0, 0, 'total_mismatch'),     # 完全不匹配
            (-1, -1, 'independent'),      # 独立选民
        ]
        
        n_per_combo = n_samples // len(combinations)
        
        for party_match, ideo_same, label in combinations:
            if party_match == -1:
                # 独立选民
                mask = self.df['party_match'] == -1
            else:
                mask = (self.df['party_match'] == party_match)
                if ideo_same != -1:
                    mask &= (self.df['ideology_same_side_7'] == ideo_same)
            
            combo_samples = self.df[mask].sample(min(n_per_combo, mask.sum()))
            
            for idx, row in combo_samples.iterrows():
                prompt = self._create_prompt_from_row(row, include_features='core')
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'party_ideology_focused',
                        'combination': label,
                        'true_score': float(row['support_score'])
                    }
                })
            
            print(f"  {label}: {len(combo_samples)} 样本")
        
        print(f"总计生成 {len(samples)} 个党派×意识形态样本")
        return samples
    
    def strategy_6_demographic_similarity(self, n_samples: int = 400) -> List[Dict]:
        """
        策略6: 基于综合相似度采样
        
        利用特征文档中的demographic_similarity综合指标
        """
        print("\n策略6: 综合相似度分层采样")
        
        samples = []
        
        # 按综合相似度分层
        similarity_bins = [
            (0.0, 0.2, 'very_low'),
            (0.2, 0.4, 'low'),
            (0.4, 0.6, 'medium'),
            (0.6, 0.8, 'high'),
            (0.8, 1.0, 'very_high')
        ]
        
        n_per_bin = n_samples // len(similarity_bins)
        
        for low, high, label in similarity_bins:
            mask = (self.df['demographic_similarity'] >= low) & \
                   (self.df['demographic_similarity'] < high)
            bin_samples = self.df[mask].sample(min(n_per_bin, mask.sum()))
            
            for idx, row in bin_samples.iterrows():
                prompt = self._create_prompt_from_row(row, include_features='full')
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'demographic_similarity',
                        'similarity_level': label,
                        'similarity_score': float(row['demographic_similarity']),
                        'true_score': float(row['support_score'])
                    }
                })
            
            print(f"  {label} ({low:.1f}-{high:.1f}): {len(bin_samples)} 样本")
        
        print(f"总计生成 {len(samples)} 个综合相似度样本")
        return samples
    
    def strategy_7_incumbent_effect(self, n_samples: int = 300) -> List[Dict]:
        """
        策略7: 现任优势效应采样
        
        专门采样现任总统/副总统，研究现任优势
        """
        print("\n策略7: 现任优势效应采样")
        
        samples = []
        
        # 现任候选人
        incumbent_mask = self.df['is_incumbent'] == 1
        incumbent_samples = self.df[incumbent_mask].sample(
            min(n_samples // 2, incumbent_mask.sum())
        )
        
        # 非现任候选人（作为对比）
        non_incumbent_mask = self.df['is_incumbent'] == 0
        non_incumbent_samples = self.df[non_incumbent_mask].sample(
            min(n_samples // 2, non_incumbent_mask.sum())
        )
        
        for df_subset, is_incumbent in [(incumbent_samples, True), 
                                        (non_incumbent_samples, False)]:
            for idx, row in df_subset.iterrows():
                prompt = self._create_prompt_from_row(row, include_features='full')
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'incumbent_effect',
                        'is_incumbent': is_incumbent,
                        'true_score': float(row['support_score'])
                    }
                })
        
        print(f"生成 {len(samples)} 个现任优势样本")
        print(f"  现任: {len(incumbent_samples)}, 非现任: {len(non_incumbent_samples)}")
        return samples
    
    def strategy_8_cross_pressures(self, n_samples: int = 300) -> List[Dict]:
        """
        策略8: 交叉压力情况采样
        
        采样那些面临"交叉压力"的选民：
        - 党派匹配但意识形态不匹配
        - 人口统计匹配但党派不匹配
        这些是最难预测的情况
        """
        print("\n策略8: 交叉压力情况采样")
        
        samples = []
        
        # 情况1: 同党派但意识形态距离大（党派忠诚 vs 意识形态分歧）
        cross_pressure_1 = (self.df['party_match'] == 1) & \
                          (self.df['ideology_distance_7'] >= 3)
        
        # 情况2: 不同党派但意识形态接近（意识形态认同 vs 党派忠诚）
        cross_pressure_2 = (self.df['party_match'] == 0) & \
                          (self.df['ideology_distance_7'] <= 1)
        
        # 情况3: 人口统计高度匹配但党派不同（身份政治 vs 党派）
        cross_pressure_3 = (self.df['demographic_similarity'] >= 0.7) & \
                          (self.df['party_match'] == 0)
        
        cross_pressures = [
            (cross_pressure_1, 'party_vs_ideology'),
            (cross_pressure_2, 'ideology_vs_party'),
            (cross_pressure_3, 'identity_vs_party')
        ]
        
        n_per_case = n_samples // len(cross_pressures)
        
        for mask, label in cross_pressures:
            case_samples = self.df[mask].sample(min(n_per_case, mask.sum()))
            
            for idx, row in case_samples.iterrows():
                prompt = self._create_prompt_from_row(row, include_features='full')
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'cross_pressures',
                        'pressure_type': label,
                        'true_score': float(row['support_score'])
                    }
                })
            
            print(f"  {label}: {len(case_samples)} 样本")
        
        print(f"总计生成 {len(samples)} 个交叉压力样本")
        return samples
    
    def strategy_9_candidate_specific(self, n_per_candidate: int = 150) -> List[Dict]:
        """
        策略5: 候选人特定样本
        
        为每个候选人生成专门的训练样本
        """
        print("\n策略5: 候选人特定采样")
        
        samples = []
        candidates = self.df['candidate'].unique()
        
        for candidate in candidates:
            cand_samples = self.df[self.df['candidate'] == candidate].sample(
                min(n_per_candidate, len(self.df[self.df['candidate'] == candidate]))
            )
            
            for idx, row in cand_samples.iterrows():
                prompt = self._create_prompt_from_row(row)
                response = self._create_response(row['support_score'])
                
                samples.append({
                    'instruction': prompt,
                    'input': '',
                    'output': response,
                    'metadata': {
                        'strategy': 'candidate_specific',
                        'candidate': candidate,
                        'true_score': float(row['support_score'])
                    }
                })
            
            print(f"  {candidate}: {len(cand_samples)} 样本")
        
        print(f"总计生成 {len(samples)} 个候选人特定样本")
        return samples
    
    def generate_full_dataset(self, output_path: str, 
                             strategies: List[str] = None,
                             model_predictions: pd.DataFrame = None):
        """
        生成完整的微调数据集
        
        Args:
            output_path: 输出文件路径
            strategies: 要使用的策略列表，None表示全部使用
            model_predictions: 模型预测结果（可选，用于难样本挖掘）
        """
        print("=" * 80)
        print("开始生成ANES微调数据集")
        print(f"数据源: {len(self.df)} 条记录")
        print("=" * 80)
        
        # 定义所有可用策略
        available_strategies = {
            'hard_cases': lambda: self.strategy_1_hard_samples(),
            'balanced': lambda: self.strategy_2_balanced_sampling(),
            'diverse': lambda: self.strategy_3_diverse_features(),
            'extreme': lambda: self.strategy_4_extreme_cases(),
            'party_ideology': lambda: self.strategy_5_party_ideology_focused(),
            'demographic_sim': lambda: self.strategy_6_demographic_similarity(),
            'incumbent': lambda: self.strategy_7_incumbent_effect(),
            'cross_pressure': lambda: self.strategy_8_cross_pressures(),
            'candidate': lambda: self.strategy_9_candidate_specific(),
        }
        
        # 如果没有指定策略，使用核心策略
        if strategies is None:
            strategies = ['balanced', 'party_ideology', 'demographic_sim', 
                         'cross_pressure', 'extreme']
            print(f"使用默认策略集: {strategies}")
        
        all_samples = []
        strategy_counts = {}
        
        # 执行选定的策略
        for strategy_name in strategies:
            if strategy_name in available_strategies:
                try:
                    samples = available_strategies[strategy_name]()
                    all_samples.extend(samples)
                    strategy_counts[strategy_name] = len(samples)
                except Exception as e:
                    print(f"❌ 策略 {strategy_name} 执行失败: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"⚠️  未知策略: {strategy_name}")
                print(f"   可用策略: {list(available_strategies.keys())}")
        
        # 去重（基于相同的prompt）
        print(f"\n去重前: {len(all_samples)} 样本")
        unique_samples = []
        seen_prompts = set()
        for sample in all_samples:
            if sample['instruction'] not in seen_prompts:
                unique_samples.append(sample)
                seen_prompts.add(sample['instruction'])
        
        print(f"去重后: {len(unique_samples)} 样本")
        
        # 打乱顺序
        np.random.shuffle(unique_samples)
        
        # 保存为JSONL格式（更适合大数据集）
        with open(output_path, 'w', encoding='utf-8') as f:
            for sample in unique_samples:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 微调数据集已保存: {output_path}")
        print(f"总样本数: {len(unique_samples)}")
        
        # 统计信息
        final_counts = {}
        for sample in unique_samples:
            strategy = sample['metadata'].get('strategy', 'unknown')
            final_counts[strategy] = final_counts.get(strategy, 0) + 1
        
        print("\n" + "=" * 80)
        print("策略分布统计:")
        print("=" * 80)
        for strategy, count in sorted(final_counts.items(), key=lambda x: -x[1]):
            percentage = count/len(unique_samples)*100
            print(f"  {strategy:20s}: {count:5d} ({percentage:5.1f}%)")
        print("=" * 80)
        
        return unique_samples


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='ANES预测任务 - 微调数据集生成')
    parser.add_argument('--data', type=str, 
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/processed/anes_all_years_with_features.csv',
                       help='输入数据路径')
    parser.add_argument('--output', type=str,
                       default='/home/jpzhao_team/cyyuan/FAS_PRO/data/anes/finetune/anes_finetune_dataset.jsonl',
                       help='输出数据路径（JSONL格式）')
    parser.add_argument('--strategies', type=str, nargs='+',
                       default=None,
                       help='使用的策略列表。可选: balanced, hard_cases, diverse, extreme, ' +
                            'party_ideology, demographic_sim, incumbent, cross_pressure, candidate')
    parser.add_argument('--all', action='store_true',
                       help='使用所有策略')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='最终采样数量（随机下采样）')
    parser.add_argument('--show-examples', type=int, default=2,
                       help='显示样本示例数量')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ANES选民支持度预测 - 微调数据集生成器")
    print("=" * 80)
    print(f"📁 输入数据: {args.data}")
    print(f"📁 输出路径: {args.output}")
    
    # 创建输出目录
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    
    # 创建生成器
    try:
        generator = ANESFinetuneDataGenerator(args.data)
    except Exception as e:
        print(f"❌ 加载数据失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 确定策略列表
    if args.all:
        strategies = ['balanced', 'hard_cases', 'diverse', 'extreme', 
                     'party_ideology', 'demographic_sim', 'incumbent', 
                     'cross_pressure', 'candidate']
        print(f"📋 使用所有策略")
    elif args.strategies:
        strategies = args.strategies
        print(f"📋 使用指定策略: {strategies}")
    else:
        strategies = None  # 使用默认策略
        print("📋 使用默认策略集（推荐）")
    
    # 生成数据集
    try:
        dataset = generator.generate_full_dataset(args.output, strategies=strategies)
    except Exception as e:
        print(f"❌ 生成数据集失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 如果指定了采样大小，进行下采样
    if args.sample_size and args.sample_size < len(dataset):
        print(f"\n🎲 随机下采样到 {args.sample_size} 个样本...")
        sampled_dataset = list(np.random.choice(dataset, args.sample_size, replace=False))
        
        # 重新保存
        with open(args.output, 'w', encoding='utf-8') as f:
            for sample in sampled_dataset:
                f.write(json.dumps(sample, ensure_ascii=False) + '\n')
        
        print(f"✅ 已保存下采样数据集")
        dataset = sampled_dataset
    
    # 显示示例
    if args.show_examples > 0:
        print("\n" + "=" * 80)
        print("📝 数据集示例:")
        print("=" * 80)
        for i, sample in enumerate(dataset[:args.show_examples]):
            print(f"\n【样本 {i+1}】")
            print(f"策略: {sample['metadata'].get('strategy', 'unknown')}")
            print(f"真实分数: {sample['metadata'].get('true_score', 'N/A')}")
            print(f"\n问题 (Instruction):")
            print(sample['instruction'])
            print(f"\n答案 (Output): {sample['output']}")
            print("-" * 80)
    
    print("\n✅ 全部完成!")
    return dataset


if __name__ == '__main__':
    main()

