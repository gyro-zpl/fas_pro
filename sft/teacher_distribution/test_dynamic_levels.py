#!/usr/bin/env python3
"""
测试动态levels功能的演示脚本
"""

import numpy as np
import pandas as pd
import sys
import os

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from get_distribution_dynamic import (
    set_levels, get_levels, generate_levels_range,
    process_data_with_attrs, get_current_distributions
)
from get_distribution_nonlinear_dynamic import (
    predict_with_dynamic_levels, demo_dynamic_levels
)

def test_basic_dynamic_levels():
    """测试基本的动态levels功能"""
    print("🧪 测试基本动态levels功能")
    print("=" * 50)
    
    # 测试1: 默认levels
    print(f"1. 默认levels: {get_levels()}")
    
    # 测试2: 设置新的levels
    new_levels = ['1', '2', '3', '4', '5']
    set_levels(new_levels)
    print(f"2. 设置新levels: {get_levels()}")
    
    # 测试3: 生成范围
    range_levels = generate_levels_range(1, 7)
    set_levels(range_levels)
    print(f"3. 生成范围1-7: {get_levels()}")
    
    # 测试4: 自定义标签
    custom_levels = ['低', '中', '高', '很高']
    set_levels(custom_levels)
    print(f"4. 自定义标签: {get_levels()}")
    
    print("✅ 基本功能测试完成\n")

def create_mock_data(levels, years=[2010, 2015, 2020]):
    """创建模拟数据用于测试"""
    np.random.seed(42)  # 确保可复现
    n_levels = len(levels)
    
    stats_by_year = {}
    for year in years:
        # 创建随机的边际分布
        pa_data = np.random.dirichlet([1] * n_levels)
        pb_data = np.random.dirichlet([1] * n_levels)
        pc_data = np.random.dirichlet([1] * n_levels)
        
        # 创建随机的联合分布
        bc_data = np.random.dirichlet([1] * (n_levels * n_levels)).reshape(n_levels, n_levels)
        bc_data = bc_data / bc_data.sum()
        
        stats_by_year[year] = {
            'P_nateduc': pd.Series(pa_data, index=levels),
            'P_natdrug': pd.Series(pb_data, index=levels),
            'P_natcrime': pd.Series(pc_data, index=levels),
            'P_natdrug_natcrime': pd.DataFrame(bc_data, index=levels, columns=levels),
            'P_nateduc_natdrug': pd.DataFrame(np.random.dirichlet([1] * (n_levels * n_levels)).reshape(n_levels, n_levels), index=levels, columns=levels),
            'P_nateduc_natcrime': pd.DataFrame(np.random.dirichlet([1] * (n_levels * n_levels)).reshape(n_levels, n_levels), index=levels, columns=levels)
        }
    
    return stats_by_year

def test_different_levels_configs():
    """测试不同的levels配置"""
    print("🧪 测试不同levels配置的预测功能")
    print("=" * 50)
    
    configs = [
        (['1', '2', '3'], "3级别配置"),
        (['1', '2', '3', '4', '5'], "5级别配置"),
        (generate_levels_range(1, 8), "1-8范围配置"),
        (['低', '中', '高'], "自定义标签配置")
    ]
    
    from get_distribution_nonlinear_dynamic import project_to_year_nonlinear
    
    for i, (levels_config, description) in enumerate(configs):
        print(f"\n--- 测试 {i+1}: {description} ---")
        print(f"Levels: {levels_config}")
        
        try:
            # 创建模拟数据
            mock_data = create_mock_data(levels_config)
            
            # 进行非线性预测
            P_a, P_b, P_c, P_bc = project_to_year_nonlinear(
                stats_by_year=mock_data,
                target_year=2025,
                attr_names=['nateduc', 'natdrug', 'natcrime'],
                levels=levels_config,
                method="auto"
            )
            
            print(f"✅ 预测成功!")
            print(f"   边际分布维度: {len(P_a)} × 3")
            print(f"   联合分布维度: {P_bc.shape}")
            print(f"   P_A (前3项): {dict(list(P_a.round(3).items())[:3])}")
            
        except Exception as e:
            print(f"❌ 预测失败: {e}")
    
    print("\n✅ 不同配置测试完成\n")

def test_dynamic_process_function():
    """测试动态处理函数"""
    print("🧪 测试动态处理函数")
    print("=" * 50)
    
    # 恢复默认levels用于真实数据测试
    set_levels(['1', '2', '3'])
    
    try:
        # 测试process_data_with_attrs函数，使用不同的levels配置
        print("测试 process_data_with_attrs 函数...")
        
        # 注意：这里可能会因为数据文件不存在而失败，这是正常的
        # 我们主要测试函数调用是否正常
        
        # 测试不同的levels_config参数格式
        test_configs = [
            (None, "默认配置"),
            (['1', '2', '3', '4'], "列表配置"),
            ((1, 5), "范围元组配置")
        ]
        
        for levels_config, description in test_configs:
            print(f"\n尝试 {description}: {levels_config}")
            try:
                # 注意：这可能会因为找不到数据文件而失败
                result = process_data_with_attrs(levels_config=levels_config)
                print(f"✅ {description} 成功")
            except FileNotFoundError:
                print(f"⚠️  {description} - 数据文件未找到（这是预期的）")
            except Exception as e:
                print(f"❌ {description} 失败: {e}")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
    
    print("✅ 动态处理函数测试完成\n")

def main():
    """主测试函数"""
    print("🚀 开始动态Levels功能测试")
    print("=" * 80)
    
    # 运行各项测试
    test_basic_dynamic_levels()
    test_different_levels_configs() 
    test_dynamic_process_function()
    
    # 运行演示函数
    print("🎯 运行非线性外推演示")
    print("=" * 50)
    demo_dynamic_levels()
    
    print("\n" + "=" * 80)
    print("🎉 所有测试完成!")
    print("\n📋 功能总结:")
    print("✅ 支持动态设置levels范围（如1-10）")
    print("✅ 支持自定义levels标签")
    print("✅ 支持不同维度的IPF计算") 
    print("✅ 支持非线性外推到任意levels配置")
    print("✅ 向后兼容原有的3级别配置")
    
    print("\n🔧 使用方法:")
    print("1. set_levels(['1','2','3','4','5']) - 设置5级别")
    print("2. set_levels(generate_levels_range(1,10)) - 设置1-10范围")
    print("3. process_data_with_attrs(levels_config=(1,8)) - 直接使用1-8范围处理")
    print("4. process_data_with_attrs(levels_config=['低','中','高']) - 使用自定义标签")
    print("=" * 80)

if __name__ == "__main__":
    main()
