"""
配额退火系统使用示例

这个脚本展示了如何使用配额退火和子群体判别系统
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from quota_annealing_system import (
    IterativeSamplingSystem, 
    QuotaManager, 
    Quota,
    QuotaAnnealing,
    SubgroupDiscriminator
)
from make_finetune_datasets import ATTR_A, ATTR_B, ATTR_C

# 设置中文显示（如果需要）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


def example_1_basic_single_quota():
    """
    示例1：基础单维度配额
    
    场景：希望属性A的值'3'占比25%
    """
    print("=" * 70)
    print("示例1：基础单维度配额")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.25, 'tolerance': 0.05, 'priority': 3}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=200,
        max_iterations=10,
        convergence_threshold=0.03,
        enable_subgroup_check=False  # 简单场景不需要子群体检查
    )
    
    print(f"\n✅ 完成! 最佳得分: {result['best_score']:.4f}")
    
    return system, result


def example_2_multi_dimension_quota():
    """
    示例2：多维度配额
    
    场景：希望"A='3'且B='2'"的组合占比10%
    """
    print("\n" + "=" * 70)
    print("示例2：多维度配额")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        {'type': 'multi', 'dimensions': {ATTR_A: '3', ATTR_B: '2'}, 
         'target': 0.10, 'tolerance': 0.03, 'priority': 4}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=300,
        max_iterations=15,
        convergence_threshold=0.02,
        enable_subgroup_check=True
    )
    
    print(f"\n✅ 完成! 最佳得分: {result['best_score']:.4f}")
    
    return system, result


def example_3_mixed_quotas_with_priority():
    """
    示例3：混合配额 + 优先级设置
    
    场景：同时控制多个配额，且有不同优先级
    """
    print("\n" + "=" * 70)
    print("示例3：混合配额 + 优先级设置")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        # 高优先级：关键业务指标
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.22, 'tolerance': 0.03, 'weight': 1.5, 'priority': 5},
        
        # 中优先级：重要指标
        {'type': 'single', 'attr': ATTR_B, 'value': '2', 
         'target': 0.35, 'tolerance': 0.05, 'weight': 1.0, 'priority': 3},
        {'type': 'multi', 'dimensions': {ATTR_A: '3', ATTR_B: '2'}, 
         'target': 0.08, 'tolerance': 0.03, 'priority': 4},
        
        # 低优先级：次要指标
        {'type': 'single', 'attr': ATTR_C, 'value': '1', 
         'target': 0.50, 'tolerance': 0.08, 'weight': 0.8, 'priority': 2}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=500,
        max_iterations=25,
        convergence_threshold=0.02,
        enable_subgroup_check=True
    )
    
    print(f"\n✅ 完成! 最佳得分: {result['best_score']:.4f}")
    
    # 详细分析
    print("\n配额满足情况:")
    for quota_name, info in result['final_quota_satisfaction'].items():
        status = "✓" if info['satisfied'] else "✗"
        print(f"  {status} {quota_name}: {info['actual']:.3f} "
              f"(目标: {info['target']:.3f}, 偏差: {info['deviation']:.3f})")
    
    return system, result


def example_4_annealing_analysis():
    """
    示例4：退火过程分析
    
    展示配额权重如何随迭代变化
    """
    print("\n" + "=" * 70)
    print("示例4：退火过程分析")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.25, 'tolerance': 0.05, 'priority': 3},
        {'type': 'single', 'attr': ATTR_B, 'value': '3', 
         'target': 0.20, 'tolerance': 0.05, 'priority': 2}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=300,
        max_iterations=20,
        convergence_threshold=0.02,
        enable_subgroup_check=True
    )
    
    # 提取退火历史
    print("\n退火历史:")
    annealing_report = system.annealing.get_annealing_report()
    
    for quota_name, perf in annealing_report['quotas_performance'].items():
        print(f"\n配额: {quota_name}")
        print(f"  总迭代: {perf['iterations']}")
        print(f"  最新偏差: {perf['latest_deviation']:.4f}")
        print(f"  是否改善: {'是' if perf['is_improving'] else '否'}")
        print(f"  可靠性得分: {perf['reliability']:.3f}")
    
    # 绘制偏差变化图
    plot_deviation_history(result['iteration_history'])
    
    return system, result


def example_5_subgroup_quality_check():
    """
    示例5：子群体质量检查
    
    重点关注子群体内部的分布质量
    """
    print("\n" + "=" * 70)
    print("示例5：子群体质量检查")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.25, 'tolerance': 0.05, 'priority': 3},
        {'type': 'single', 'attr': ATTR_B, 'value': '2', 
         'target': 0.35, 'tolerance': 0.05, 'priority': 3}
    ]
    
    system.set_quotas(quota_configs)
    
    # 自定义质量阈值
    system.discriminator.quality_thresholds = {
        'diversity': 0.35,
        'concentration': 0.75,
        'balance': 0.25
    }
    
    result = system.run_iteration(
        n_samples=400,
        max_iterations=20,
        convergence_threshold=0.02,
        enable_subgroup_check=True
    )
    
    # 分析子群体质量
    print("\n子群体质量分析:")
    last_iteration = result['iteration_history'][-1]
    if last_iteration['subgroup_report']:
        sr = last_iteration['subgroup_report']
        print(f"平均质量得分: {sr['average_quality']:.3f}")
        print(f"通过率: {sr['pass_rate']:.2%}")
        print(f"总体通过: {'是' if sr['overall_passes'] else '否'}")
        
        print("\n各子群体详情:")
        for subgroup_name, info in sr['subgroups'].items():
            print(f"\n  {subgroup_name}:")
            print(f"    样本量: {info['size']}")
            print(f"    比例: {info['proportion']:.3f}")
            print(f"    质量得分: {info['quality_score']:.3f}")
            print(f"    多样性: {info['diversity']:.3f}")
            print(f"    集中度: {info['concentration']:.3f}")
            if info['issues']:
                print(f"    问题: {', '.join(info['issues'])}")
    
    return system, result


def example_6_realistic_scenario():
    """
    示例6：真实场景应用
    
    模拟实际业务需求：生成符合特定人口统计学分布的样本
    """
    print("\n" + "=" * 70)
    print("示例6：真实场景应用 - 人口统计学配额")
    print("=" * 70)
    
    system = IterativeSamplingSystem()
    
    # 模拟真实的人口统计学配额需求
    quota_configs = [
        # 边际配额：模拟年龄、性别、教育程度分布
        {'type': 'single', 'attr': ATTR_A, 'value': '1', 
         'target': 0.55, 'tolerance': 0.05, 'priority': 4, 
         'weight': 1.2},  # 年龄段1占55%
        {'type': 'single', 'attr': ATTR_A, 'value': '2', 
         'target': 0.30, 'tolerance': 0.05, 'priority': 3},
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.15, 'tolerance': 0.04, 'priority': 3},
        
        # 交叉配额：特定组合
        {'type': 'multi', 'dimensions': {ATTR_A: '1', ATTR_B: '2'}, 
         'target': 0.20, 'tolerance': 0.04, 'priority': 5},  # 关键组合
        {'type': 'multi', 'dimensions': {ATTR_A: '2', ATTR_C: '3'}, 
         'target': 0.08, 'tolerance': 0.03, 'priority': 3}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=1000,  # 较大样本量
        max_iterations=30,
        convergence_threshold=0.015,  # 严格收敛
        enable_subgroup_check=True
    )
    
    print(f"\n✅ 生成完成!")
    print(f"最佳质量得分: {result['best_score']:.4f}")
    print(f"收敛状态: {'收敛' if result['convergence_achieved'] else '未完全收敛'}")
    print(f"总迭代次数: {result['total_iterations']}")
    
    # 导出结果
    system.save_results('/home/cyyuan/FAS_PRO/sft/teacher_distribution/realistic_scenario_report.json')
    system.export_best_samples('/home/cyyuan/FAS_PRO/sft/teacher_distribution/realistic_samples.csv')
    
    print("\n结果已保存到:")
    print("  - realistic_scenario_report.json")
    print("  - realistic_samples.csv")
    
    return system, result


def plot_deviation_history(iteration_history):
    """绘制偏差变化历史"""
    try:
        iterations = [h['iteration'] for h in iteration_history]
        deviations = [h['total_deviation'] for h in iteration_history]
        temperatures = [h['temperature'] for h in iteration_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        
        # 偏差变化
        ax1.plot(iterations, deviations, marker='o', linewidth=2, markersize=6)
        ax1.set_xlabel('迭代次数', fontsize=12)
        ax1.set_ylabel('总偏差', fontsize=12)
        ax1.set_title('配额偏差随迭代的变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.02, color='r', linestyle='--', label='收敛阈值')
        ax1.legend()
        
        # 温度变化
        ax2.plot(iterations, temperatures, marker='s', color='orange', linewidth=2, markersize=6)
        ax2.set_xlabel('迭代次数', fontsize=12)
        ax2.set_ylabel('退火温度', fontsize=12)
        ax2.set_title('退火温度变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/home/cyyuan/FAS_PRO/sft/teacher_distribution/annealing_history.png', dpi=150)
        print("\n📊 图表已保存: annealing_history.png")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def plot_quota_comparison(result):
    """绘制配额目标vs实际对比图"""
    try:
        quota_data = result['final_quota_satisfaction']
        
        quota_names = list(quota_data.keys())
        targets = [info['target'] for info in quota_data.values()]
        actuals = [info['actual'] for info in quota_data.values()]
        satisfied = [info['satisfied'] for info in quota_data.values()]
        
        x = np.arange(len(quota_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['green' if s else 'red' for s in satisfied]
        bars1 = ax.bar(x - width/2, targets, width, label='目标', alpha=0.7)
        bars2 = ax.bar(x + width/2, actuals, width, label='实际', alpha=0.7, color=colors)
        
        ax.set_xlabel('配额', fontsize=12)
        ax.set_ylabel('比例', fontsize=12)
        ax.set_title('配额目标 vs 实际达成对比', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(quota_names, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('/home/cyyuan/FAS_PRO/sft/teacher_distribution/quota_comparison.png', dpi=150)
        print("📊 图表已保存: quota_comparison.png")
        
    except Exception as e:
        print(f"绘图失败: {e}")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "🚀" * 35)
    print("配额退火系统 - 完整示例演示")
    print("🚀" * 35)
    
    # 运行所有示例
    examples = [
        ("基础单维度配额", example_1_basic_single_quota),
        ("多维度配额", example_2_multi_dimension_quota),
        ("混合配额+优先级", example_3_mixed_quotas_with_priority),
        ("退火过程分析", example_4_annealing_analysis),
        ("子群体质量检查", example_5_subgroup_quality_check),
        ("真实场景应用", example_6_realistic_scenario)
    ]
    
    results = {}
    
    for name, example_func in examples:
        try:
            print(f"\n{'=' * 70}")
            print(f"正在运行: {name}")
            print('=' * 70)
            system, result = example_func()
            results[name] = (system, result)
            print(f"✅ {name} 完成!")
        except Exception as e:
            print(f"❌ {name} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 生成对比报告
    print("\n" + "=" * 70)
    print("汇总报告")
    print("=" * 70)
    
    comparison_data = []
    for name, (system, result) in results.items():
        comparison_data.append({
            '示例': name,
            '最佳得分': f"{result['best_score']:.4f}",
            '迭代次数': result['total_iterations'],
            '收敛': '是' if result['convergence_achieved'] else '否',
            '样本数': result['best_samples_count']
        })
    
    df_comparison = pd.DataFrame(comparison_data)
    print("\n" + df_comparison.to_string(index=False))
    
    return results


def quick_test():
    """快速测试：验证系统基本功能"""
    print("\n" + "⚡" * 35)
    print("快速测试")
    print("⚡" * 35)
    
    system = IterativeSamplingSystem()
    
    quota_configs = [
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 
         'target': 0.25, 'tolerance': 0.05, 'priority': 3}
    ]
    
    system.set_quotas(quota_configs)
    
    result = system.run_iteration(
        n_samples=100,  # 小样本快速测试
        max_iterations=5,
        convergence_threshold=0.05,
        enable_subgroup_check=False
    )
    
    print(f"\n✅ 快速测试完成!")
    print(f"得分: {result['best_score']:.4f}")
    print(f"迭代: {result['total_iterations']}")
    
    return system, result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='配额退火系统示例')
    parser.add_argument('--mode', type=str, default='quick', 
                       choices=['quick', 'all', '1', '2', '3', '4', '5', '6'],
                       help='运行模式: quick(快速测试), all(所有示例), 1-6(特定示例)')
    
    args = parser.parse_args()
    
    if args.mode == 'quick':
        quick_test()
    elif args.mode == 'all':
        run_all_examples()
    elif args.mode == '1':
        example_1_basic_single_quota()
    elif args.mode == '2':
        example_2_multi_dimension_quota()
    elif args.mode == '3':
        system, result = example_3_mixed_quotas_with_priority()
        plot_quota_comparison(result)
    elif args.mode == '4':
        example_4_annealing_analysis()
    elif args.mode == '5':
        example_5_subgroup_quality_check()
    elif args.mode == '6':
        example_6_realistic_scenario()
    
    print("\n" + "🎉" * 35)
    print("运行完成!")
    print("🎉" * 35)

