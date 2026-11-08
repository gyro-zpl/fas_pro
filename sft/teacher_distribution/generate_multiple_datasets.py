#!/usr/bin/env python3
"""
多属性数据集生成脚本
无需修改源码，支持任意属性组合的微调数据集生成
"""

import sys
sys.path.append('/home/cyyuan/FAS_PRO/sft/teacher_distribution')

from make_finetune_datasets import *
import pandas as pd
import numpy as np
import json

# =========================
# 多种属性组合配置
# =========================

# 配置1：原始的城市/环境/健康
CONFIG_1 = {
    "natcity": "national spending on solving the problems of big cities",
    "natenvir": "national spending on improving and protecting the environment", 
    "natheal": "national spending on improving and protecting the nation's health"
}

# 配置2：犯罪/毒品/教育
CONFIG_2 = {
    
    "nateduc": "national spending on education",
    "natdrug": "national spending on dealing with drug addiction", 
    "natcrime": "national spending on halting the rising crime rate",
}

# 配置3：社会福利相关
CONFIG_3 = {
    "natfare": "national spending on welfare",
    "natsoc": "national spending on social security", 
    "nataid": "national spending on foreign aid"
}

# 配置4：科技/环境/公园
CONFIG_4 = {
    "natscience": "national spending on scientific research",
    "natenvir": "national spending on improving and protecting the environment",
    "natpark": "national spending on parks and recreation"
}

# =========================
# 配置验证和便捷函数
# 注意：分布参数会由 make_finetune_datasets.py 自动从真实数据计算
# =========================

def validate_config(config_name):
    """
    验证配置名称是否有效
    """
    valid_configs = ["config_1", "config_2", "config_3", "config_4"]
    if config_name not in valid_configs:
        raise ValueError(f"未知的配置名称: {config_name}. 可用配置: {valid_configs}")
    return True

def generate_single_config(config_name, total_samples=2000):
    """
    生成单个配置的数据集（便捷函数）
    
    Args:
        config_name: 配置名称 ("config_1", "config_2", "config_3", "config_4")
        total_samples: 样本数量
    """
    config_map = {
        "config_1": (CONFIG_1, "城市/环境/健康"),
        "config_2": (CONFIG_2, "犯罪/毒品/教育"),
        "config_3": (CONFIG_3, "社会福利相关"),
        "config_4": (CONFIG_4, "科技/环境/公园")
    }
    
    if config_name not in config_map:
        raise ValueError(f"未知配置: {config_name}. 可用: {list(config_map.keys())}")
    
    config_dict, description = config_map[config_name]
    print(f"🎯 生成单个配置: {config_name} ({description})")
    
    return generate_dataset_for_config(
        config_dict, config_name, 
        total_samples=total_samples,
        output_prefix="single_config_dataset"
    )

def generate_dataset_for_config(config_dict, config_name, total_samples=1000, output_prefix="dataset"):
    """
    为特定属性配置生成微调数据集
    
    Args:
        config_dict: 属性配置字典
        config_name: 配置名称
        total_samples: 总样本数
        output_prefix: 输出文件前缀
    """
    print(f"\n=== 生成 {config_name} 数据集 ===")
    
    # 验证配置
    validate_config(config_name)
    
    # 1. 更新属性配置
    update_attr_config(config_dict)
    
    # 2. 生成数据集
    # 注意：generate_balanced_finetune_dataset 会自动初始化真实分布数据
    output_file = f"{output_prefix}_{config_name}_{total_samples}.json"
    dataset = generate_balanced_finetune_dataset(
        total_samples=total_samples,
        output_file=output_file,
        use_realistic_priors=True
    )
    
    print(f"✅ {config_name} 数据集生成完成: {output_file}")
    return dataset

def main():
    """生成多个属性配置的数据集"""
    
    # 所有可用的配置
    all_configs = [
        (CONFIG_1, "config_1", "城市/环境/健康"),
        (CONFIG_2, "config_2", "犯罪/毒品/教育"),
        (CONFIG_3, "config_3", "社会福利相关"),
        (CONFIG_4, "config_4", "科技/环境/公园")
    ]
    
    # 默认使用前两个配置（可以修改）
    active_configs = all_configs[:2]  # 修改这里来选择要生成的配置
    
    # 推荐的数据集规模
    recommended_sizes = {
        "tiny": 100,       # 快速测试
        "small": 500,      # 小规模测试
        "medium": 2000,    # 标准训练
        "large": 5000,     # 充分训练
        "xlarge": 10000    # 大规模训练
    }
    
    size = "medium"  # 修改这里选择规模
    total_samples = recommended_sizes[size]
    
    print(f"🎯 将为 {len(active_configs)} 种属性配置各生成 {total_samples} 个样本")
    print("📋 活跃配置:")
    for _, config_name, description in active_configs:
        print(f"  - {config_name}: {description}")
    print(f"📏 数据集规模: {size} ({total_samples} 样本/配置)")
    
    all_datasets = []
    for config_dict, config_name, description in active_configs:
        print(f"\n--- 处理 {description} ---")
        try:
            dataset = generate_dataset_for_config(
                config_dict, config_name, 
                total_samples=total_samples, 
                output_prefix="multiattr_dataset"
            )
            all_datasets.extend(dataset)
        except Exception as e:
            print(f"❌ {config_name} 生成失败: {e}")
            continue
    
    if all_datasets:
        # 合并所有数据集
        combined_file = f"/home/cyyuan/FAS_PRO/sft/JSON_data/combined_multiattr_dataset_{len(all_datasets)}.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_datasets, f, ensure_ascii=False, indent=2)
        
        print(f"\n🎉 所有数据集生成完成！")
        print(f"📊 总计样本数: {len(all_datasets)}")
        print(f"📁 合并文件: {combined_file}")
        print(f"📁 单独文件: /home/cyyuan/FAS_PRO/sft/JSON_data/multiattr_dataset_config_*_{total_samples}.json")
    else:
        print(f"\n❌ 没有成功生成任何数据集")

if __name__ == "__main__":
    main()
