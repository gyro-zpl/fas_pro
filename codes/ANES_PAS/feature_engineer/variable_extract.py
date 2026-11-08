#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSV列提取工具
从CSV文件中提取指定的列并保存到新文件
"""

import pandas as pd
import os
import sys


def extract_columns(input_file, output_file, columns):
    """
    从CSV文件中提取指定的列
    
    参数:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        columns: 要提取的列名列表
    """
    try:
        # 读取CSV文件
        print(f"正在读取文件: {input_file}")
        df = pd.read_csv(input_file)
        
        # 显示原始数据信息
        #print(f"原始数据形状: {df.shape}")
        #print(f"原始数据列名: {df.columns.tolist()}")
        
        # 检查哪些列存在
        existing_columns = [col for col in columns if col in df.columns]
        missing_columns = [col for col in columns if col not in df.columns]
        
        if missing_columns:
            print(f"\n警告: 以下列在数据中不存在: {missing_columns}")
        
        if not existing_columns:
            print("错误: 没有找到任何指定的列!")
            return False
        
        # 提取存在的列
        print(f"\n提取以下列: {existing_columns}")
        df_extracted = df[existing_columns]
        
        # 保存到新文件
        print(f"正在保存到: {output_file}")
        df_extracted.to_csv(output_file, index=False)
        
        print(f"\n成功! 提取后数据形状: {df_extracted.shape}")
        print(f"提取的列数: {len(existing_columns)}")
        print(f"数据行数: {len(df_extracted)}")
        
        # 显示前几行数据预览
        print("\n数据预览:")
        print(df_extracted.head())
        
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
        return False
    except Exception as e:
        print(f"错误: {str(e)}")
        return False


if __name__ == "__main__":
    # 定义要提取的列
    columns_to_extract = [
        'birthyr', 
        'gender', 
        'race', 
        'educ', 
        'marstat', 
        'employ', 
        'faminc', 
        'votereg',
        'ftobama',
        'ftblack',
        'ftwhite',
        'fttrump',
        'fthrc',
        'ftcarson',
        'ftrubio',
        'ftcruz',
        'ftsanders',
        'ftfiorina',
        'ftjeb',
        'pid7',
        'ideo5',
        'lcself',
        'fthrc',
        'ftobama',
        'ftkavanaugh',
        'ftmueller',
        'ftbiden',
        'ftwarren',
        'ftbuttigieg',
        'ftharris'
        'ftklobuchar',
        'ftpence1',
        'ftyang1',
        'ftpelosi1',
        'ftrubio1',
        'ftocasioc1',
        'fthaley1',
        'ftthomas1',
        'ftfauci1',
        'age',
        'home_ownership',
        'income',
        'marital1',
        'race7',
        'cand_trump',
        'cand_biden',
        'group_demsoc'
        



    ]
    
    # 方式1: 通过命令行参数指定输入输出文件
    if len(sys.argv) >= 3:
        input_csv = sys.argv[1]
        output_csv = sys.argv[2]
    # 方式2: 在代码中直接指定（修改下面的路径）
    else:
        # 请修改这里的文件路径
        input_csv = "input.csv"  # 输入文件路径
        output_csv = "output_extracted.csv"  # 输出文件路径
        
        print("使用默认文件路径。")
        print(f"如需指定文件，请使用: python {sys.argv[0]} <输入文件> <输出文件>")
        print()
    
    # 执行提取
    success = extract_columns(input_csv, output_csv, columns_to_extract)
    
    if success:
        print("\n处理完成!")
    else:
        print("\n处理失败!")
        sys.exit(1)

# python variable_extract.py /home/cyyuan/FAS_PRO/data/anes/anes_pilot_2016.csv /home/cyyuan/FAS_PRO/data/anes/anes_pilot_2016_extracted.csv