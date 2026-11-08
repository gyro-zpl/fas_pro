#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
创建受访者×候选人交互特征（改进版）
计算差异特征：意识形态差异、年龄差异、党派差异、种族差异等

特点：
1. 灵活处理不同年份的不同属性
2. 标准化列名映射（自动适配不同命名方式）
3. 丰富的交互特征工程
4. 完善的缺失值处理
5. 特征质量检查和报告

作者: CY Yuan
日期: 2025-10-16
"""

import pandas as pd
import numpy as np
import os
import warnings
from typing import Dict, List, Tuple, Optional
warnings.filterwarnings('ignore')


# ==================== 配置部分 ====================

# 列名标准化映射（处理不同年份的命名差异）
COLUMN_MAPPING = {
    # 种族相关
    'race7': 'race',
    'race5': 'race',
    'race_resp': 'race',
    
    # 婚姻状况
    'marital1': 'marstat',
    'marital': 'marstat',
    
    # 收入相关
    'income': 'faminc',
    'family_income': 'faminc',
    
    # 就业状况
    'employment': 'employ',
    'emp': 'employ',
    
    # 年龄（有些数据集直接提供age）
    'age_resp': 'age',
    
    # 性别
    'sex': 'gender',
    
    # 党派认同
    'pid3': 'party_id_3',
    'partyid': 'pid7',
    
    # 意识形态
    'ideology': 'ideo5',
    'ideology5': 'ideo5',
    'ideology7': 'lcself',
}

# 候选人元数据必需列
CANDIDATE_REQUIRED_COLS = ['candidate', 'year', 'party', 'ideology']

# 候选人元数据可选列
CANDIDATE_OPTIONAL_COLS = ['birthyr', 'race', 'gender', 'position']


def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    标准化列名，处理不同年份的命名差异
    
    参数:
        df: 原始数据框
    
    返回:
        df: 标准化列名后的数据框
    """
    df = df.copy()
    
    # 应用列名映射
    rename_dict = {}
    for col in df.columns:
        if col in COLUMN_MAPPING:
            rename_dict[col] = COLUMN_MAPPING[col]
    
    if rename_dict:
        df.rename(columns=rename_dict, inplace=True)
        print(f"  列名标准化: {rename_dict}")
    
    return df


def load_candidate_metadata(metadata_file: str) -> pd.DataFrame:
    """
    加载候选人元数据
    
    参数:
        metadata_file: 元数据文件路径
    
    返回:
        df_meta: 候选人元数据
    """
    print(f"加载候选人元数据: {metadata_file}")
    df_meta = pd.read_csv(metadata_file)
    
    # 检查必需列
    missing_cols = set(CANDIDATE_REQUIRED_COLS) - set(df_meta.columns)
    if missing_cols:
        raise ValueError(f"候选人元数据缺少必需列: {missing_cols}")
    
    print(f"候选人元数据形状: {df_meta.shape}")
    print(f"包含年份: {sorted(df_meta['year'].unique())}")
    print(f"包含候选人: {len(df_meta['candidate'].unique())} 个")
    
    # 打印可用属性
    available_attrs = [col for col in CANDIDATE_OPTIONAL_COLS if col in df_meta.columns]
    print(f"可用属性: {available_attrs}")
    
    return df_meta


def check_data_quality(df: pd.DataFrame, data_type: str = "数据") -> None:
    """
    检查数据质量并打印报告
    
    参数:
        df: 数据框
        data_type: 数据类型描述
    """
    print(f"\n{data_type}质量检查:")
    print(f"  样本数: {len(df)}")
    print(f"  特征数: {len(df.columns)}")
    
    # 缺失值统计
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_info = pd.DataFrame({
        '缺失数': missing,
        '缺失率(%)': missing_pct
    })
    missing_info = missing_info[missing_info['缺失数'] > 0].sort_values('缺失率(%)', ascending=False)
    
    if len(missing_info) > 0:
        print(f"\n  缺失值情况（前10）:")
        print(missing_info.head(10).to_string())
    else:
        print("  无缺失值")


def prepare_merge_columns(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """
    准备合并前的列名处理，避免冲突
    
    参数:
        df: 数据框
        suffix: 后缀（'_resp' 或 '_cand'）
    
    返回:
        df: 处理后的数据框
    """
    df = df.copy()
    
    # 需要重命名的列（可能与候选人/受访者属性冲突）
    overlap_cols = ['birthyr', 'gender', 'race', 'ideology']
    
    for col in overlap_cols:
        if col in df.columns:
            df[f'{col}{suffix}'] = df[col]
            df.drop(col, axis=1, inplace=True)
    
    return df


def calculate_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算年龄相关特征
    
    参数:
        df: 包含年龄信息的数据框
    
    返回:
        df: 添加年龄特征后的数据框
    """
    df = df.copy()
    
    # 计算受访者年龄
    if 'birthyr_resp' in df.columns and 'year' in df.columns:
        df['age_resp'] = df['year'] - df['birthyr_resp']
    elif 'age' in df.columns:
        # 如果直接提供了age列，重命名为age_resp
        df['age_resp'] = df['age']
    
    # 计算候选人年龄
    if 'birthyr_cand' in df.columns and 'year' in df.columns:
        df['age_cand'] = df['year'] - df['birthyr_cand']
    
    # 年龄差异
    if 'age_resp' in df.columns and 'age_cand' in df.columns:
        df['age_difference'] = np.abs(df['age_resp'] - df['age_cand'])
        
        # 年龄差异比例（标准化）
        df['age_difference_ratio'] = df['age_difference'] / df['age_resp'].replace(0, np.nan)
        
        # 年龄代差分类
        df['age_gap_category'] = pd.cut(
            df['age_difference'],
            bins=[0, 15, 30, 100],
            labels=['same_gen', 'adjacent_gen', 'cross_gen']
        ).astype(str)
        
        # 受访者是否年长（处理NaN）
        df['resp_older'] = (df['age_resp'] > df['age_cand']).fillna(0).astype(int)
        
        # 年龄段匹配（按10年分组，处理NaN）
        # 先计算，然后填充NaN，最后转换为int
        df['age_decade_resp'] = (df['age_resp'] // 10).fillna(-1)
        df['age_decade_cand'] = (df['age_cand'] // 10).fillna(-1)
        
        # 将浮点数转换为int（-1表示缺失）
        df['age_decade_resp'] = df['age_decade_resp'].astype(int)
        df['age_decade_cand'] = df['age_decade_cand'].astype(int)
        
        # 计算差异（如果任一方是-1，结果也是-1）
        df['age_decade_diff'] = np.where(
            (df['age_decade_resp'] == -1) | (df['age_decade_cand'] == -1),
            -1,
            np.abs(df['age_decade_resp'] - df['age_decade_cand'])
        )
        
        print(f"  ✓ 年龄特征: 创建了 8 个特征")
    else:
        print(f"  ✗ 年龄特征: 缺少必要的年龄信息")
    
    return df


def calculate_party_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算党派相关特征
    
    参数:
        df: 包含党派信息的数据框
    
    返回:
        df: 添加党派特征后的数据框
    """
    df = df.copy()
    
    if 'pid7' not in df.columns or 'party' not in df.columns:
        print(f"  ✗ 党派特征: 缺少 pid7 或 party 列")
        return df
    
    # 受访者党派倾向（1=民主党, 2=共和党, 0=独立）
    df['party_alignment'] = 0
    df.loc[df['pid7'] <= 3, 'party_alignment'] = 1  # 民主党倾向
    df.loc[df['pid7'] >= 5, 'party_alignment'] = 2  # 共和党倾向
    df.loc[df['pid7'] == 4, 'party_alignment'] = 0  # 独立选民
    
    # 党派强度（距离中间派的距离）
    df['party_strength_resp'] = np.abs(df['pid7'] - 4)  # 0-3
    
    # 党派匹配
    df['party_match'] = (df['party_alignment'] == df['party']).astype(int)
    df.loc[df['party_alignment'] == 0, 'party_match'] = -1  # 独立选民标记为-1
    
    # 同党派二值
    df['same_party'] = (df['party_alignment'] == df['party']).astype(int)
    
    # 党派极性（受访者和候选人是否处于相反阵营）
    df['opposite_party'] = (
        ((df['party_alignment'] == 1) & (df['party'] == 2)) |
        ((df['party_alignment'] == 2) & (df['party'] == 1))
    ).astype(int)
    
    # 将候选人党派映射到pid7量表
    df['pid7_cand'] = df['party'].map({1: 2, 2: 6})  # 民主党=2, 共和党=6
    df['party_distance'] = np.abs(df['pid7'] - df['pid7_cand'])
    
    # 党派强度交互
    df['party_strength_x_match'] = df['party_strength_resp'] * df['same_party']
    df['party_strength_x_distance'] = df['party_strength_resp'] * df['party_distance']
    
    print(f"  ✓ 党派特征: 创建了 9 个特征")
    
    return df


def calculate_ideology_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算意识形态相关特征
    
    参数:
        df: 包含意识形态信息的数据框
    
    返回:
        df: 添加意识形态特征后的数据框
    """
    df = df.copy()
    
    if 'ideology_cand' not in df.columns:
        print(f"  ✗ 意识形态特征: 缺少候选人意识形态信息")
        return df
    
    feature_count = 0
    
    # 将候选人ideology (1-7) 映射到5分制
    df['ideology_cand_5scale'] = ((df['ideology_cand'] - 1) * 4 / 6 + 1).round()
    feature_count += 1
    
    # 使用ideo5计算差异（5分制）
    if 'ideo5' in df.columns:
        df['ideology_distance_5'] = np.abs(df['ideo5'] - df['ideology_cand_5scale'])
        df['ideology_same_side_5'] = (
            ((df['ideo5'] <= 3) & (df['ideology_cand_5scale'] <= 3)) |  # 都偏自由
            ((df['ideo5'] >= 3) & (df['ideology_cand_5scale'] >= 3))     # 都偏保守
        ).astype(int)
        feature_count += 2
    
    # 使用lcself计算差异（7分制）
    if 'lcself' in df.columns:
        df['ideology_distance_7'] = np.abs(df['lcself'] - df['ideology_cand'])
        df['ideology_same_side_7'] = (
            ((df['lcself'] <= 4) & (df['ideology_cand'] <= 4)) |  # 都偏自由
            ((df['lcself'] >= 4) & (df['ideology_cand'] >= 4))    # 都偏保守
        ).astype(int)
        
        # 意识形态极性匹配（都在极端vs都在中间）
        df['both_extreme_ideology'] = (
            ((df['lcself'] <= 2) | (df['lcself'] >= 6)) & 
            ((df['ideology_cand'] <= 2) | (df['ideology_cand'] >= 6))
        ).astype(int)
        
        df['both_moderate_ideology'] = (
            ((df['lcself'] >= 3) & (df['lcself'] <= 5)) & 
            ((df['ideology_cand'] >= 3) & (df['ideology_cand'] <= 5))
        ).astype(int)
        
        # 意识形态强度
        df['ideology_strength_resp'] = np.abs(df['lcself'] - 4)
        df['ideology_strength_cand'] = np.abs(df['ideology_cand'] - 4)
        
        feature_count += 6
    
    print(f"  ✓ 意识形态特征: 创建了 {feature_count} 个特征")
    
    return df


def calculate_demographic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算人口统计特征（种族、性别等）
    
    参数:
        df: 包含人口统计信息的数据框
    
    返回:
        df: 添加人口统计特征后的数据框
    """
    df = df.copy()
    feature_count = 0
    
    # 种族特征
    if 'race_resp' in df.columns and 'race_cand' in df.columns:
        df['race_match'] = (df['race_resp'] == df['race_cand']).astype(int)
        
        # 是否同为白人
        df['both_white'] = ((df['race_resp'] == 1) & (df['race_cand'] == 1)).astype(int)
    
    # 是否同为少数族裔
        df['both_minority'] = ((df['race_resp'] != 1) & (df['race_cand'] != 1)).astype(int)
    
        # 跨种族组合
        df['cross_race'] = (df['race_resp'] != df['race_cand']).astype(int)
        
        feature_count += 4
    
    # 性别特征
    if 'gender_resp' in df.columns and 'gender_cand' in df.columns:
        df['gender_match'] = (df['gender_resp'] == df['gender_cand']).astype(int)
        
        # 女性受访者评价男性候选人
        df['female_resp_male_cand'] = (
            (df['gender_resp'] == 2) & (df['gender_cand'] == 1)
        ).astype(int)
        
        # 男性受访者评价女性候选人
        df['male_resp_female_cand'] = (
            (df['gender_resp'] == 1) & (df['gender_cand'] == 2)
        ).astype(int)
        
        # 同性别组合
        df['same_gender'] = (df['gender_resp'] == df['gender_cand']).astype(int)
        
        feature_count += 4
    
    print(f"  ✓ 人口统计特征: 创建了 {feature_count} 个特征")
    
    return df


def calculate_compound_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算复合交互特征（二阶、三阶交互）
    
    参数:
        df: 包含基础特征的数据框
    
    返回:
        df: 添加复合特征后的数据框
    """
    df = df.copy()
    feature_count = 0
    
    # 党派 × 意识形态
    if 'party_match' in df.columns and 'ideology_distance_7' in df.columns:
        df['party_match_x_ideology_dist'] = df['party_match'] * df['ideology_distance_7']
        df['party_same_x_ideology_same'] = df.get('same_party', 0) * df.get('ideology_same_side_7', 0)
        feature_count += 2
    
    # 种族 × 党派
    if 'race_match' in df.columns and 'party_match' in df.columns:
        df['race_match_x_party_match'] = df['race_match'] * df['party_match']
        feature_count += 1
    
    # 性别 × 党派
    if 'gender_match' in df.columns and 'party_match' in df.columns:
        df['gender_match_x_party_match'] = df['gender_match'] * df['party_match']
        feature_count += 1
    
    # 年龄 × 教育
    if 'age_difference' in df.columns and 'educ' in df.columns:
        df['age_diff_x_educ'] = df['age_difference'] * df['educ']
        feature_count += 1
    
    # 年龄 × 党派
    if 'age_difference' in df.columns and 'party_distance' in df.columns:
        df['age_diff_x_party_dist'] = df['age_difference'] * df['party_distance']
        feature_count += 1
    
    # 教育 × 意识形态
    if 'educ' in df.columns and 'ideology_distance_7' in df.columns:
        df['educ_x_ideology_dist'] = df['educ'] * df['ideology_distance_7']
        feature_count += 1
    
    # 收入 × 党派
    if 'faminc' in df.columns and 'party_distance' in df.columns:
        df['income_x_party_dist'] = df['faminc'] * df['party_distance']
        feature_count += 1
    
    # 三阶交互：党派 × 意识形态 × 种族
    if all(col in df.columns for col in ['party_match', 'ideology_same_side_7', 'race_match']):
        df['party_ideology_race_align'] = (
            df['party_match'] * df['ideology_same_side_7'] * df['race_match']
        )
        feature_count += 1
    
    # 综合相似度得分（多维度加权）
    similarity_score = 0
    weights_sum = 0
    
    if 'race_match' in df.columns:
        similarity_score += df['race_match'] * 0.2
        weights_sum += 0.2
    
    if 'gender_match' in df.columns:
        similarity_score += df['gender_match'] * 0.15
        weights_sum += 0.15
    
    if 'age_difference' in df.columns:
        age_similarity = (1 - df['age_difference'] / 100).clip(0, 1)
        similarity_score += age_similarity * 0.15
        weights_sum += 0.15
    
    if 'party_match' in df.columns:
        party_similarity = (df['party_match'] + 1) / 2  # 转换 -1,0,1 到 0,0.5,1
        similarity_score += party_similarity * 0.25
        weights_sum += 0.25
    
    if 'ideology_distance_7' in df.columns:
        ideology_similarity = (1 - df['ideology_distance_7'] / 6).clip(0, 1)
        similarity_score += ideology_similarity * 0.25
        weights_sum += 0.25
    
    if weights_sum > 0:
        df['demographic_similarity'] = similarity_score / weights_sum
        feature_count += 1
    
    print(f"  ✓ 复合交互特征: 创建了 {feature_count} 个特征")
    
    return df


def calculate_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    计算情境特征（年份、职位等）
    
    参数:
        df: 包含情境信息的数据框
    
    返回:
        df: 添加情境特征后的数据框
    """
    df = df.copy()
    feature_count = 0
    
    # 年份相关
    if 'year' in df.columns:
        # 选举年标记
        df['is_election_year'] = (df['year'] % 2 == 0).astype(int)
        
        # 总统选举年
        df['is_presidential_year'] = (df['year'] % 4 == 0).astype(int)
        
        # 年份归一化（时间趋势）
        year_min = df['year'].min()
        year_max = df['year'].max()
        if year_max > year_min:
            df['year_normalized'] = (df['year'] - year_min) / (year_max - year_min)
        else:
            df['year_normalized'] = 0
        
        feature_count += 3
    
    # 候选人职位编码
    if 'position' in df.columns:
        position_map = {
        'president': 1,
        'vice_president': 2,
        'candidate': 3,
        'former_president': 4,
        'house_speaker': 5,
        'senator': 6,
        'governor': 7,
        'judge': 8,
        'doctor': 9,
        'special_counsel': 10,
        'representative': 11
        }
        df['position_encoded'] = df['position'].map(position_map).fillna(0).astype(int)
    
        # 是否现任总统/副总统
        df['is_incumbent'] = df['position'].isin(['president', 'vice_president']).astype(int)
    
        # 是否候选人
        df['is_candidate'] = (df['position'] == 'candidate').astype(int)
    
        feature_count += 3
    
    print(f"  ✓ 情境特征: 创建了 {feature_count} 个特征")
    
    return df


def create_interaction_features(df_long: pd.DataFrame, 
                                df_candidate_meta: pd.DataFrame) -> pd.DataFrame:
    """
    创建受访者×候选人的交互特征（主函数）
    
    参数:
        df_long: 长表格式的数据
        df_candidate_meta: 候选人元数据
    
    返回:
        df_features: 包含交互特征的数据框
    """
    print("\n" + "="*60)
    print("创建交互特征")
    print("="*60)
    
    # 1. 标准化列名
    print("\n步骤1: 标准化列名")
    df_long = standardize_column_names(df_long)
    
    # 2. 准备合并
    print("\n步骤2: 准备数据合并")
    df_long_prep = prepare_merge_columns(df_long, '_resp')
    df_meta_prep = prepare_merge_columns(df_candidate_meta, '_cand')
    
    # 3. 合并候选人元数据
    print("\n步骤3: 合并候选人元数据")
    df = df_long_prep.merge(
        df_meta_prep,
        on=['candidate', 'year'],
        how='left',
        suffixes=('_resp', '_cand')
    )
    
    print(f"  合并后数据形状: {df.shape}")
    
    # 检查缺失的候选人元数据
    missing_meta = df[df['party'].isna()]
    if len(missing_meta) > 0:
        print(f"  ⚠ 警告: {len(missing_meta)} 条记录缺少候选人元数据")
        print("  缺失元数据的候选人:")
        print(missing_meta.groupby(['year', 'candidate']).size().head(10))
    
    # 4. 转换数值型列
    print("\n步骤4: 转换数值型列")
    numeric_cols = ['pid7', 'ideo5', 'lcself', 'educ', 'marstat', 'employ', 'faminc', 'votereg']
    converted_count = 0
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            converted_count += 1
    print(f"  转换了 {converted_count} 个数值列")
    
    # 5. 创建各类特征
    print("\n步骤5: 创建交互特征")
    
    df = calculate_age_features(df)
    df = calculate_party_features(df)
    df = calculate_ideology_features(df)
    df = calculate_demographic_features(df)
    df = calculate_compound_features(df)
    df = calculate_contextual_features(df)
    
    # 6. 质量检查
    original_cols = set(df_long.columns)
    new_cols = set(df.columns) - original_cols
    print(f"\n步骤6: 特征创建完成")
    print(f"  原始列数: {len(df_long.columns)}")
    print(f"  新增列数: {len(new_cols)}")
    print(f"  总列数: {len(df.columns)}")
    
    check_data_quality(df, "合并后数据")
    
    return df


def get_feature_categories() -> Dict[str, List[str]]:
    """
    获取特征分类
    
    返回:
        feature_categories: 特征分类字典
    """
    feature_categories = {
        # 受访者基础特征
        'respondent_basic': [
            'age_resp', 'gender_resp', 'race_resp', 'educ', 'marstat', 
            'employ', 'faminc', 'votereg', 'birthyr_resp'
        ],
        
        # 受访者政治倾向特征
        'respondent_political': [
            'pid7', 'ideo5', 'lcself', 'party_alignment', 'party_strength_resp'
        ],
        
        # 候选人基础特征
        'candidate_basic': [
            'party', 'ideology_cand', 'age_cand', 'race_cand', 
            'gender_cand', 'birthyr_cand'
        ],
        
        # 候选人情境特征
        'candidate_contextual': [
            'position_encoded', 'is_incumbent', 'is_candidate'
        ],
        
        # 年龄交互特征
        'age_interaction': [
            'age_difference', 'age_difference_ratio', 'age_gap_category',
            'resp_older', 'age_decade_resp', 'age_decade_cand', 'age_decade_diff'
        ],
        
        # 党派交互特征
        'party_interaction': [
            'party_match', 'same_party', 'opposite_party',
            'party_distance', 'pid7_cand',
            'party_strength_x_match', 'party_strength_x_distance'
        ],
        
        # 意识形态交互特征
        'ideology_interaction': [
            'ideology_cand_5scale', 'ideology_distance_5', 'ideology_distance_7',
            'ideology_same_side_5', 'ideology_same_side_7',
            'both_extreme_ideology', 'both_moderate_ideology',
            'ideology_strength_resp', 'ideology_strength_cand'
        ],
        
        # 人口统计交互特征
        'demographic_interaction': [
            'race_match', 'both_white', 'both_minority', 'cross_race',
            'gender_match', 'female_resp_male_cand', 'male_resp_female_cand', 
            'same_gender'
        ],
        
        # 复合交互特征
        'compound_interaction': [
            'party_match_x_ideology_dist', 'party_same_x_ideology_same',
            'race_match_x_party_match', 'gender_match_x_party_match',
            'age_diff_x_educ', 'age_diff_x_party_dist',
            'educ_x_ideology_dist', 'income_x_party_dist',
            'party_ideology_race_align', 'demographic_similarity'
        ],
        
        # 年份情境特征
        'year_contextual': [
            'year', 'is_election_year', 'is_presidential_year', 'year_normalized'
        ]
    }
    
    return feature_categories


def prepare_training_data(df: pd.DataFrame, 
                          target_type: str = 'classification',
                          include_meta_info: bool = False) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    准备训练数据（改进版，更灵活地处理不同年份的特征）
    
    参数:
        df: 包含所有特征的数据框
        target_type: 'regression' 或 'classification'
        include_meta_info: 是否包含元信息（respondent_id, year, candidate等）
    
    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
        df_valid: 有效数据（包含元信息）
    """
    print(f"\n" + "="*60)
    print(f"准备训练数据（目标类型: {target_type}）")
    print("="*60)
    
    # 1. 过滤有效数据
    if 'is_missing' in df.columns:
        df_valid = df[df['is_missing'] == 0].copy()
        print(f"\n有效数据数量: {len(df_valid)} / {len(df)} ({len(df_valid)/len(df)*100:.1f}%)")
    else:
        # 如果没有is_missing列，过滤有support_score的数据
        df_valid = df[df['support_score'].notna()].copy()
        print(f"\n有效数据数量: {len(df_valid)} / {len(df)} ({len(df_valid)/len(df)*100:.1f}%)")
    
    # 2. 获取特征分类
    feature_categories = get_feature_categories()
    
    # 3. 收集所有可用特征
    all_desired_features = []
    for category_name, features in feature_categories.items():
        all_desired_features.extend(features)
    
    # 4. 过滤实际存在的特征
    available_features = [f for f in all_desired_features if f in df_valid.columns]
    missing_features = set(all_desired_features) - set(available_features)
    
    print(f"\n期望特征数: {len(all_desired_features)}")
    print(f"可用特征数: {len(available_features)}")
    
    if missing_features:
        print(f"\n缺失特征 ({len(missing_features)}个):")
        for feat in sorted(missing_features)[:20]:  # 只显示前20个
            print(f"  - {feat}")
        if len(missing_features) > 20:
            print(f"  ... 还有 {len(missing_features) - 20} 个")
    
    # 5. 按类别统计特征
    print(f"\n按类别统计可用特征:")
    for category_name, features in feature_categories.items():
        available_in_category = [f for f in features if f in available_features]
        if available_in_category:
            print(f"  {category_name}: {len(available_in_category)}/{len(features)} 个")
    
    # 6. 提取特征
    X = df_valid[available_features].copy()
    
    # 7. 处理分类变量（one-hot编码）
    categorical_cols = []
    if 'age_gap_category' in X.columns:
        categorical_cols.append('age_gap_category')
    
    if categorical_cols:
        print(f"\n对分类变量进行One-Hot编码: {categorical_cols}")
        X = pd.get_dummies(X, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    
    # 8. 处理缺失值
    print(f"\n处理缺失值:")
    missing_count_before = X.isnull().sum().sum()
    print(f"  缺失值总数（处理前）: {missing_count_before}")
    
    for col in X.columns:
        if X[col].isna().any():
            if X[col].dtype in ['float64', 'int64']:
                # 数值型：用中位数填充
                median_val = X[col].median()
                X[col].fillna(median_val, inplace=True)
            else:
                # 分类型：用众数或'unknown'填充
                X[col].fillna('unknown', inplace=True)
    
    missing_count_after = X.isnull().sum().sum()
    print(f"  缺失值总数（处理后）: {missing_count_after}")
    
    # 9. 准备目标变量
    print(f"\n准备目标变量:")
    
    # 首先确保 support_score 是数值型
    df_valid['support_score'] = pd.to_numeric(df_valid['support_score'], errors='coerce')
    
    # 移除 support_score 为 NaN 的行
    df_valid_with_target = df_valid[df_valid['support_score'].notna()].copy()
    if len(df_valid_with_target) < len(df_valid):
        print(f"  移除了 {len(df_valid) - len(df_valid_with_target)} 个无效目标值的样本")
        df_valid = df_valid_with_target
        # 同步更新特征矩阵
        X = X.loc[df_valid.index]
    
    if target_type == 'classification':
        # 将支持度离散化为等级
        # 0-20: 0(强烈反对), 21-40: 1(反对), 41-60: 2(中立), 
        # 61-80: 3(支持), 81-100: 4(强烈支持)
        
        # 先裁剪到有效范围
        support_clipped = df_valid['support_score'].clip(0, 100)
        
        y = pd.cut(
            support_clipped,
            bins=[-0.001, 20, 40, 60, 80, 100.001],  # 扩大上界以包含100
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        )
        
        # 处理任何可能的NaN值
        if y.isna().any():
            print(f"  警告: {y.isna().sum()} 个样本无法分类，将被移除")
            valid_idx = ~y.isna()
            y = y[valid_idx]
            X = X.loc[valid_idx]
            df_valid = df_valid.loc[valid_idx]
        
        y = y.astype(int)
        
        print(f"  分类标签分布:")
        class_dist = y.value_counts().sort_index()
        for class_label, count in class_dist.items():
            print(f"    类别 {class_label}: {count} ({count/len(y)*100:.1f}%)")
    else:
        # 回归任务
        y = df_valid['support_score'].copy()
        print(f"  回归目标统计:")
        print(f"    均值: {y.mean():.2f}")
        print(f"    标准差: {y.std():.2f}")
        print(f"    最小值: {y.min():.2f}")
        print(f"    最大值: {y.max():.2f}")
    
    # 10. 保存元信息（如果需要）
    if include_meta_info:
        meta_cols = ['respondent_id', 'year', 'candidate']
        available_meta = [c for c in meta_cols if c in df_valid.columns]
        if available_meta:
            for col in available_meta:
                df_valid[f'meta_{col}'] = df_valid[col]
    
    print(f"\n" + "="*60)
    print(f"训练数据准备完成")
    print(f"  特征矩阵形状: {X.shape}")
    print(f"  目标变量形状: {y.shape}")
    print("="*60)
    
    return X, y, X.columns.tolist(), df_valid


def save_processed_data(df: pd.DataFrame, output_file: str) -> None:
    """
    保存处理后的数据
    
    参数:
        df: 数据框
        output_file: 输出文件路径
    """
    print(f"\n保存处理后的数据到: {output_file}")
    df.to_csv(output_file, index=False)
    print(f"  保存完成，数据形状: {df.shape}")
    print(f"  文件大小: {os.path.getsize(output_file) / (1024*1024):.2f} MB")


def export_feature_summary(feature_names: List[str], output_file: str) -> None:
    """
    导出特征摘要
    
    参数:
        feature_names: 特征名称列表
        output_file: 输出文件路径
    """
    # 获取特征分类
    feature_categories = get_feature_categories()
    
    # 创建特征分类映射
    feature_to_category = {}
    for category, features in feature_categories.items():
        for feat in features:
            feature_to_category[feat] = category
    
    # 创建特征摘要
    summary_data = []
    for feat in feature_names:
        category = feature_to_category.get(feat, 'other')
        summary_data.append({
            'feature_name': feat,
            'category': category
        })
    
    df_summary = pd.DataFrame(summary_data)
    df_summary.to_csv(output_file, index=False)
    print(f"\n特征摘要已保存到: {output_file}")
    print(f"  总特征数: {len(feature_names)}")
    print(f"\n按类别统计:")
    cat_counts = df_summary['category'].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} 个")


def main(data_dir: Optional[str] = None, 
         years: Optional[List[int]] = None,
         output_dir: Optional[str] = None,
         target_type: str = 'classification') -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame], 
                                                        Optional[pd.Series], Optional[List[str]]]:
    """
    主函数：处理所有年份的数据，创建交互特征（改进版）
    
    参数:
        data_dir: 数据目录路径
        years: 要处理的年份列表
        output_dir: 输出目录路径
        target_type: 目标变量类型 ('classification' 或 'regression')
    
    返回:
        df_all: 合并的所有数据
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名称列表
    """
    print("="*80)
    print(" "*20 + "受访者×候选人交互特征生成器")
    print("="*80)
    print(f"版本: 2.0")
    print(f"日期: 2025-10-16")
    print("="*80)
    
    # 1. 路径配置
    if data_dir is None:
        data_dir = '/home/cyyuan/FAS_PRO/data/anes'
    
    if years is None:
        # 自动检测可用年份
        years = []
        for year in [2016, 2018, 2019, 2020, 2021, 2022, 2024]:
            input_file = os.path.join(data_dir, f'anes_pilot_{year}_long.csv')
            if os.path.exists(input_file):
                years.append(year)
        print(f"\n检测到可用年份: {years}")
    
    if output_dir is None:
        output_dir = os.path.join(data_dir, 'processed')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"数据目录: {data_dir}")
    print(f"输出目录: {output_dir}")
    
    # 2. 加载候选人元数据
    print(f"\n" + "="*80)
    print("步骤1: 加载候选人元数据")
    print("="*80)
    
    candidate_meta_file = os.path.join(data_dir, 'candidate_metadata.csv')
    if not os.path.exists(candidate_meta_file):
        print(f"错误: 候选人元数据文件不存在 {candidate_meta_file}")
        return None, None, None, None
    
    df_candidate_meta = load_candidate_metadata(candidate_meta_file)
    
    # 3. 处理每个年份
    print(f"\n" + "="*80)
    print("步骤2: 处理各年份数据")
    print("="*80)
    
    all_data = []
    processing_summary = []
    
    for year in years:
        print(f"\n{'-'*80}")
        print(f"处理 {year} 年数据")
        print(f"{'-'*80}")
        
        input_file = os.path.join(data_dir, f'anes_pilot_{year}_long.csv')
        
        if not os.path.exists(input_file):
            print(f"⚠ 警告: 文件不存在 {input_file}，跳过")
            processing_summary.append({
                'year': year,
                'status': 'skipped',
                'rows': 0,
                'reason': '文件不存在'
            })
            continue
        
        
        # 加载长表数据
        try:
            print(f"\n加载数据: {input_file}")
            df_long = pd.read_csv(input_file)
            print(f"  原始数据形状: {df_long.shape}")
            print(f"  列名: {list(df_long.columns[:10])}{'...' if len(df_long.columns) > 10 else ''}")
        
        # 清理候选人名称（去掉数字后缀，如pelosi1 -> pelosi）
            if 'candidate' in df_long.columns:
                df_long['candidate'] = df_long['candidate'].str.replace(r'\d+$', '', regex=True)
        
        # 创建交互特征
            df_with_features = create_interaction_features(df_long, df_candidate_meta)
            
            # 保存单年数据
            output_file = os.path.join(output_dir, f'anes_{year}_with_features.csv')
            save_processed_data(df_with_features, output_file)
            
            all_data.append(df_with_features)
    
            processing_summary.append({
                    'year': year,
                    'status': 'success',
                    'rows': len(df_with_features),
                    'cols': len(df_with_features.columns),
                    'reason': '处理成功'
                })
            
        except Exception as e:
            print(f"✗ 错误: 处理 {year} 年数据时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            processing_summary.append({
                'year': year,
                'status': 'error',
                'rows': 0,
                'reason': str(e)
            })
    
    # 4. 打印处理摘要
    print(f"\n" + "="*80)
    print("处理摘要")
    print("="*80)
    df_summary = pd.DataFrame(processing_summary)
    print(df_summary.to_string(index=False))
    
    # 5. 合并所有年份
    if not all_data:
        print("\n错误: 没有成功处理任何数据")
        return None, None, None, None
    
    print(f"\n" + "="*80)
    print("步骤3: 合并所有年份数据")
    print("="*80)
        
        df_all = pd.concat(all_data, axis=0, ignore_index=True)
    print(f"\n合并后数据形状: {df_all.shape}")
    print(f"包含年份: {sorted(df_all['year'].unique())}")
    print(f"包含候选人数: {len(df_all['candidate'].unique())}")
        
        # 保存合并数据
        merged_file = os.path.join(output_dir, 'anes_all_years_with_features.csv')
        save_processed_data(df_all, merged_file)
        
    # 6. 准备训练数据
    print(f"\n" + "="*80)
    print("步骤4: 准备训练数据")
    print("="*80)
    
    X, y, feature_names, df_valid = prepare_training_data(
        df_all, 
        target_type=target_type,
        include_meta_info=True
    )
    
    # 7. 保存训练数据
    print(f"\n" + "="*80)
    print("步骤5: 保存训练数据")
    print("="*80)
    
    X_file = os.path.join(output_dir, f'X_features_{target_type}.csv')
    y_file = os.path.join(output_dir, f'y_target_{target_type}.csv')
    feature_summary_file = os.path.join(output_dir, 'feature_summary.csv')
    
    print(f"\n保存特征矩阵到: {X_file}")
    X.to_csv(X_file, index=False)
    
    print(f"保存目标变量到: {y_file}")
    y.to_frame('target').to_csv(y_file, index=False)
    
    # 导出特征摘要
    export_feature_summary(feature_names, feature_summary_file)
    
    # 8. 最终报告
    print(f"\n" + "="*80)
    print("✓ 处理完成！")
    print("="*80)
    
    print(f"\n输出文件:")
    print(f"  1. 单年特征文件:")
    for year in years:
        file_path = os.path.join(output_dir, f'anes_{year}_with_features.csv')
        if os.path.exists(file_path):
            size_mb = os.path.getsize(file_path) / (1024*1024)
            print(f"     - anes_{year}_with_features.csv ({size_mb:.2f} MB)")
    
    print(f"\n  2. 合并数据文件:")
    print(f"     - anes_all_years_with_features.csv")
    
    print(f"\n  3. 训练数据文件:")
    print(f"     - X_features_{target_type}.csv (特征矩阵: {X.shape})")
    print(f"     - y_target_{target_type}.csv (目标变量: {y.shape})")
    print(f"     - feature_summary.csv (特征摘要: {len(feature_names)} 个特征)")
    
    print(f"\n数据统计:")
    print(f"  总样本数: {len(df_all)}")
    print(f"  有效样本数: {len(df_valid)}")
    print(f"  总特征数: {len(feature_names)}")
    print(f"  候选人数: {len(df_all['candidate'].unique())}")
    print(f"  年份范围: {df_all['year'].min()} - {df_all['year'].max()}")
    
    print(f"\n" + "="*80)
    
    return df_all, X, y, feature_names


if __name__ == '__main__':
    df_all, X, y, feature_names = main()

