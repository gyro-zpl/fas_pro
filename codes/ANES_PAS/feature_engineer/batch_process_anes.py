#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量处理多年ANES数据
"""

from anes_wide_to_long import wide_to_long_anes
import pandas as pd
import os


def batch_process_anes_data():
    """
    批量处理多个年份的ANES数据，并合并为一个文件
    """
    # 数据目录
    data_dir = '/home/cyyuan/FAS_PRO/data/anes'
    
    # 定义要处理的文件（根据实际情况修改）
    input_files = {
        2016: os.path.join(data_dir, 'anes_pilot_2016.csv'),
        # 如果有其他年份，继续添加：
        # 2017: os.path.join(data_dir, 'anes_pilot_2017.csv'),
        # 2018: os.path.join(data_dir, 'anes_pilot_2018.csv'),
        # 2019: os.path.join(data_dir, 'anes_pilot_2019.csv'),
        # 2020: os.path.join(data_dir, 'anes_pilot_2020.csv'),
    }
    
    # 输出文件
    output_dir = os.path.join(data_dir, 'long_format')
    os.makedirs(output_dir, exist_ok=True)
    
    all_data = []
    
    # 处理每个年份
    for year, input_file in input_files.items():
        print(f"\n{'='*60}")
        print(f"处理 {year} 年数据")
        print(f"{'='*60}")
        
        if not os.path.exists(input_file):
            print(f"警告: 文件不存在 {input_file}，跳过")
            continue
        
        # 输出文件路径
        output_file = os.path.join(output_dir, f'anes_pilot_{year}_long.csv')
        
        # 转换
        success = wide_to_long_anes(input_file, output_file, year=year)
        
        if success:
            # 读取转换后的数据
            df = pd.read_csv(output_file)
            all_data.append(df)
            print(f"✓ {year} 年数据处理成功，共 {len(df)} 行")
        else:
            print(f"✗ {year} 年数据处理失败")
    
    # 合并所有年份
    if all_data:
        print(f"\n{'='*60}")
        print("合并所有年份数据")
        print(f"{'='*60}")
        
        merged_df = pd.concat(all_data, axis=0, ignore_index=True)
        
        # 重新生成全局受访者ID（跨年份唯一）
        # 格式：year_originalid
        merged_df['global_respondent_id'] = (
            merged_df['year'].astype(str) + '_' + 
            merged_df['respondent_id'].astype(str)
        )
        
        # 保存合并结果
        merged_output = os.path.join(output_dir, 'anes_pilot_all_years_long.csv')
        merged_df.to_csv(merged_output, index=False)
        
        # 统计信息
        print(f"\n合并完成！")
        print(f"总行数: {len(merged_df):,}")
        print(f"包含年份: {sorted(merged_df['year'].unique())}")
        print(f"受访者数量（按年份）:")
        for year in sorted(merged_df['year'].unique()):
            n = merged_df[merged_df['year'] == year]['respondent_id'].nunique()
            print(f"  {year}: {n:,} 人")
        
        print(f"\n候选人/对象数量（按年份）:")
        for year in sorted(merged_df['year'].unique()):
            candidates = merged_df[merged_df['year'] == year]['candidate'].unique()
            print(f"  {year}: {len(candidates)} 个")
            print(f"    {', '.join(sorted(candidates))}")
        
        print(f"\n所有文件已保存到: {output_dir}")
        print(f"合并文件: {merged_output}")
        
        # 数据质量报告
        print(f"\n{'='*60}")
        print("数据质量报告")
        print(f"{'='*60}")
        
        total_missing = merged_df['is_missing'].sum()
        total_records = len(merged_df)
        missing_rate = (total_missing / total_records) * 100
        
        print(f"总记录数: {total_records:,}")
        print(f"缺失记录数: {total_missing:,}")
        print(f"缺失率: {missing_rate:.2f}%")
        
        # 按候选人统计缺失率
        print(f"\n候选人支持度数据完整性（前10个）:")
        candidate_stats = merged_df.groupby('candidate').agg({
            'is_missing': 'mean',
            'support_score': 'count'
        }).sort_values('is_missing')
        candidate_stats.columns = ['缺失率', '记录数']
        candidate_stats['缺失率'] = (candidate_stats['缺失率'] * 100).round(2)
        print(candidate_stats.head(10))
        
        return merged_df
    else:
        print("\n错误: 没有成功处理任何数据文件")
        return None


if __name__ == '__main__':
    df = batch_process_anes_data()
    
    if df is not None:
        print("\n处理完成！")
    else:
        print("\n处理失败！")

