#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ANES数据宽表转长表工具
将不同年份的ANES数据从宽表格式转换为长表格式
"""

import pandas as pd
import numpy as np
import sys
import os


def wide_to_long_anes(input_file, output_file, year=None):
    """
    将ANES宽表数据转换为长表格式
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        year: 数据年份（如果未指定，将从文件名推断）
    """
    try:
        # 读取数据
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        print(f"原始数据形状: {df.shape}")
        
        # 推断年份
        if year is None:
            # 尝试从文件名中提取年份
            import re
            match = re.search(r'(\d{4})', os.path.basename(input_file))
            if match:
                year = int(match.group(1))
                print(f"从文件名推断年份: {year}")
            else:
                year = 9999  # 未知年份
                print("警告: 无法推断年份，使用默认值9999")
        
        # 添加受访者ID（如果不存在）
        if 'respondent_id' not in df.columns:
            df['respondent_id'] = df.index
        
        # 识别候选人支持度列（ft开头或cand_开头的列）
        ft_columns = [col for col in df.columns if col.startswith('ft')]
        cand_columns = [col for col in df.columns if col.startswith('cand_')]
        
        # 合并两种格式
        candidate_columns = ft_columns + cand_columns
        
        print(f"\n找到 {len(candidate_columns)} 个候选人/对象支持度列:")
        if ft_columns:
            print(f"  ft格式: {ft_columns}")
        if cand_columns:
            print(f"  cand格式: {cand_columns}")
        
        # 检查是否找到候选人列
        if not candidate_columns:
            print("\n错误: 未找到任何候选人支持度列（ft*或cand_*）")
            print("数据列名:", df.columns.tolist())
            return False
        
        # 识别受访者特征列（非候选人列，非respondent_id）
        # 包括人口统计信息和党派意向
        respondent_cols = [col for col in df.columns 
                          if col not in candidate_columns 
                          and col != 'respondent_id']
        print(f"\n识别到 {len(respondent_cols)} 个受访者特征列:")
        print(respondent_cols)
        
        # 转换为长表
        print("\n开始转换为长表格式...")
        rows = []
        
        for cand_col in candidate_columns:
            # 提取候选人/对象名称
            if cand_col.startswith('ft'):
                candidate_name = cand_col[2:]  # 去掉'ft'前缀
            elif cand_col.startswith('cand_'):
                candidate_name = cand_col[5:]  # 去掉'cand_'前缀
            else:
                candidate_name = cand_col
            
            # 创建临时数据框
            temp_df = df[['respondent_id'] + respondent_cols].copy()
            temp_df['candidate'] = candidate_name
            temp_df['year'] = year
            
            # 转换支持度为数值（处理"not asked"等文本）
            support_values = pd.to_numeric(df[cand_col], errors='coerce')
            temp_df['support_score'] = support_values
            
            # 添加缺失值标记
            # 包括：NaN, 997, 998, 以及无法转换为数字的值
            is_missing = (
                support_values.isna() | 
                support_values.isin([997, 998, -1, -9])
            )
            temp_df['is_missing'] = is_missing.astype(int)
            
            # 将特殊值设为NaN
            mask = temp_df['support_score'].isin([997, 998, -1, -9])
            temp_df.loc[mask, 'support_score'] = np.nan
            
            rows.append(temp_df)
        
        # 合并所有行
        df_long = pd.concat(rows, axis=0, ignore_index=True)
        
        # 重新排列列顺序
        cols_order = ['respondent_id', 'year', 'candidate', 'support_score', 'is_missing'] + respondent_cols
        df_long = df_long[cols_order]
        
        # 保存结果
        print(f"\n正在保存到: {output_file}")
        df_long.to_csv(output_file, index=False)
        
        # 显示统计信息
        print(f"\n转换完成!")
        print(f"长表数据形状: {df_long.shape}")
        print(f"受访者数量: {df_long['respondent_id'].nunique()}")
        print(f"候选人/对象数量: {df_long['candidate'].nunique()}")
        print(f"年份: {df_long['year'].unique()}")
        print(f"\n候选人/对象列表:")
        print(df_long['candidate'].unique())
        
        # 缺失值统计
        missing_rate = df_long['is_missing'].mean() * 100
        print(f"\n支持度缺失率: {missing_rate:.2f}%")
        
        # 显示前几行
        print("\n数据预览:")
        print(df_long.head(10))
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def merge_multiple_years(input_files, output_file):
    """
    合并多个年份的长表数据
    
    参数:
        input_files: 字典，格式为 {year: file_path}
        output_file: 输出文件路径
    """
    all_data = []
    
    for year, file_path in input_files.items():
        print(f"\n处理 {year} 年数据...")
        temp_output = file_path.replace('.csv', '_long.csv')
        
        if wide_to_long_anes(file_path, temp_output, year=year):
            df = pd.read_csv(temp_output)
            all_data.append(df)
            print(f"{year} 年数据处理完成，共 {len(df)} 行")
    
    if all_data:
        print("\n合并所有年份数据...")
        merged_df = pd.concat(all_data, axis=0, ignore_index=True)
        merged_df.to_csv(output_file, index=False)
        
        print(f"\n合并完成！")
        print(f"合并后数据形状: {merged_df.shape}")
        print(f"包含年份: {sorted(merged_df['year'].unique())}")
        print(f"总受访者数: {merged_df['respondent_id'].nunique()}")
        print(f"所有候选人/对象: {merged_df['candidate'].nunique()}")
        
        return True
    else:
        print("错误: 没有成功处理任何数据文件")
        return False


if __name__ == "__main__":
    # 方式1: 批量处理多个年份（示例 - 取消注释以使用）
    for year in [2016, 2018, 2019, 2020, 2022, 2024]:
        input_file = f'/home/cyyuan/FAS_PRO/data/anes/raw_data/anes_pilot_{year}_extracted.csv'
        output_file = f'/home/cyyuan/FAS_PRO/data/anes/anes_pilot_{year}_long.csv'
        success = wide_to_long_anes(input_file, output_file, year)
        if not success:
            print(f"错误: 处理 {year} 年数据失败")
        else:
            print(f"成功处理 {year} 年数据")
    
    sys.exit(0)
    # 方式2: 命令行参数（单个文件转换）
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
        year = int(sys.argv[3]) if len(sys.argv) >= 4 else None
        
        success = wide_to_long_anes(input_csv, output_csv, year)
        sys.exit(0 if success else 1)
    
    # 方式2: 批量处理多个年份（示例）
    else:
        print("用法:")
        print("  单个文件: python anes_wide_to_long.py <输入文件> <输出文件> [年份]")
        print("  示例: python anes_wide_to_long.py anes_pilot_2016.csv anes_pilot_2016_long.csv 2016")
        print("\n或者修改脚本中的文件路径进行批量处理")
        
        # 示例：批量处理多年数据（需要根据实际情况修改路径）
        """
        input_files = {
            2016: '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_2016.csv',
            2017: '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_2017.csv',
            2018: '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_2018.csv',
            2019: '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_2019.csv',
            2020: '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_2020.csv',
        }
        output_file = '/home/cyyuan/FAS_PRO/data/anes/anes_pilot_all_years_long.csv'
        merge_multiple_years(input_files, output_file)
        """
        
        sys.exit(0)

