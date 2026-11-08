#!/usr/bin/env python3
"""
比较线性预测和非线性预测的效果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from get_distribution import (load_and_process_data, year_stats, ATTR_NAMES, levels, 
                             marginal_from_Q, ipf_joint, inject_scenario, normalize)
from get_distribution_nonlinear import project_to_year_nonlinear
from plot_distribution_trends import extract_real_2024_data

def linear_project(years, values, target_year):
    """线性外推函数"""
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    if len(np.unique(years)) < 2:
        return float(values.mean())
    A = np.vstack([years, np.ones_like(years)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    pred = a * target_year + b
    return float(np.clip(pred, 0.0, 1.0))

def project_to_year_linear(stats_by_year, target_year, attr_names=None):
    """线性外推到目标年份"""
    from get_distribution import ATTR_A, ATTR_B, ATTR_C
    
    if attr_names is None:
        attr_a, attr_b, attr_c = ATTR_A, ATTR_B, ATTR_C
    else:
        attr_a, attr_b, attr_c = attr_names
    years = sorted(stats_by_year.keys())
    
    P_a = pd.Series(index=levels, dtype=float)
    P_b = pd.Series(index=levels, dtype=float)
    P_c = pd.Series(index=levels, dtype=float)
    
    for lvl in levels:
        P_a[lvl] = linear_project(years, [stats_by_year[y][f"P_{attr_a}"][lvl] for y in years], target_year)
        P_b[lvl] = linear_project(years, [stats_by_year[y][f"P_{attr_b}"][lvl] for y in years], target_year)
        P_c[lvl] = linear_project(years, [stats_by_year[y][f"P_{attr_c}"][lvl] for y in years], target_year)
    
    P_a, P_b, P_c = normalize(P_a), normalize(P_b), normalize(P_c)

    # 选用 B-C 二阶交叉来承载相关性
    bc_frames = np.vstack([stats_by_year[y][f"P_{attr_b}_{attr_c}"].values.flatten() for y in years])
    pred_bc = []
    for cell in range(9):
        pred_bc.append(linear_project(years, bc_frames[:, cell], target_year))
    pred_bc = np.array(pred_bc)
    pred_bc = pred_bc / pred_bc.sum() if pred_bc.sum() > 0 else pred_bc
    P_bc = pd.DataFrame(pred_bc.reshape(3,3), index=levels, columns=levels)

    return P_a, P_b, P_c, P_bc

def compare_prediction_methods(attr_names=None):
    """
    比较线性和非线性预测方法的效果
    """
    from get_distribution import ATTR_A, ATTR_B, ATTR_C
    
    if attr_names is None:
        attr_a, attr_b, attr_c = ATTR_A, ATTR_B, ATTR_C
    else:
        attr_a, attr_b, attr_c = attr_names
    
    # 加载数据
    print("正在加载历史数据...")
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    
    # 获取真实2024年数据
    print("正在加载2024年真实数据...")
    real_2024_stats = extract_real_2024_data()
    
    # 线性预测
    print("正在进行线性预测...")
    P_a_lin, P_b_lin, P_c_lin, P_bc_lin = project_to_year_linear(stats_by_year, 2024, None if attr_names is None else attr_names)
    P_b_lin_adj = inject_scenario(P_b_lin, delta_3=0.00)
    Q_joint_lin = ipf_joint(P_a_lin, P_b_lin_adj, P_c_lin, P_bc_lin, iters=300)
    Pa_pred_lin, Pb_pred_lin, Pc_pred_lin = marginal_from_Q(Q_joint_lin)
    
    # 非线性预测
    print("正在进行非线性预测...")
    P_a_nl, P_b_nl, P_c_nl, P_bc_nl = project_to_year_nonlinear(stats_by_year, 2024, None if attr_names is None else attr_names)
    P_b_nl_adj = inject_scenario(P_b_nl, delta_3=0.00)
    Q_joint_nl = ipf_joint(P_a_nl, P_b_nl_adj, P_c_nl, P_bc_nl, iters=300)
    Pa_pred_nl, Pb_pred_nl, Pc_pred_nl = marginal_from_Q(Q_joint_nl)
    
    # 整理预测结果
    linear_preds = {f"P_{attr_a}": Pa_pred_lin, f"P_{attr_b}": Pb_pred_lin, f"P_{attr_c}": Pc_pred_lin}
    nonlinear_preds = {f"P_{attr_a}": Pa_pred_nl, f"P_{attr_b}": Pb_pred_nl, f"P_{attr_c}": Pc_pred_nl}
    
    # 创建对比图表
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Linear vs Nonlinear vs Real: 2024 Prediction Comparison', fontsize=16, fontweight='bold')
    
    methods = ['Linear', 'Nonlinear', 'Real']
    colors = ['#1f77b4', '#ff7f0e', '#d62728']  # 蓝色、橙色、红色
    
    for var_idx, var_name in enumerate([attr_a, attr_b, attr_c]):
        ax = axes[var_idx]
        
        x = np.arange(len(levels))
        width = 0.25
        
        # 获取各方法的预测值
        linear_vals = [linear_preds[f"P_{var_name}"][level] for level in levels]
        nonlinear_vals = [nonlinear_preds[f"P_{var_name}"][level] for level in levels]
        real_vals = [real_2024_stats[f"P_{var_name}"][level] for level in levels]
        
        # 绘制柱状图
        bars1 = ax.bar(x - width, linear_vals, width, label='Linear', color=colors[0], alpha=0.8)
        bars2 = ax.bar(x, nonlinear_vals, width, label='Nonlinear', color=colors[1], alpha=0.8)
        bars3 = ax.bar(x + width, real_vals, width, label='Real', color=colors[2], alpha=0.8)
        
        # 添加数值标签
        for bars in [bars1, bars2, bars3]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}', xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Level')
        ax.set_ylabel('Probability')
        ax.set_title(f'{var_name.upper()}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Level {l}' for l in levels])
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, max(max(linear_vals), max(nonlinear_vals), max(real_vals)) * 1.1)
    
    plt.tight_layout()
    plt.savefig('/home/cyyuan/FAS_PRO/sft/graphs/prediction_methods_comparison2.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 计算预测误差
    print("\n" + "="*80)
    print("🎯 预测方法对比报告")
    print("="*80)
    
    total_mae_linear = 0
    total_mae_nonlinear = 0
    
    for var_name in [attr_a, attr_b, attr_c]:
        print(f"\n📊 {var_name.upper()} 预测对比:")
        print("-" * 50)
        
        var_mae_linear = 0
        var_mae_nonlinear = 0
        
        for level in levels:
            linear_pred = linear_preds[f"P_{var_name}"][level]
            nonlinear_pred = nonlinear_preds[f"P_{var_name}"][level]
            real_val = real_2024_stats[f"P_{var_name}"][level]
            
            linear_error = abs(real_val - linear_pred)
            nonlinear_error = abs(real_val - nonlinear_pred)
            
            var_mae_linear += linear_error
            var_mae_nonlinear += nonlinear_error
            
            # 哪个方法更好
            better = "🟢 非线性" if nonlinear_error < linear_error else "🔵 线性"
            if abs(nonlinear_error - linear_error) < 0.001:
                better = "🟡 相近"
            
            print(f"  Level {level}:")
            print(f"    真实值: {real_val:.4f}")
            print(f"    线性预测: {linear_pred:.4f} (误差: {linear_error:.4f})")
            print(f"    非线性预测: {nonlinear_pred:.4f} (误差: {nonlinear_error:.4f})")
            print(f"    更优方法: {better}")
        
        var_mae_linear /= len(levels)
        var_mae_nonlinear /= len(levels)
        
        print(f"\n  平均绝对误差:")
        print(f"    线性: {var_mae_linear:.4f}")
        print(f"    非线性: {var_mae_nonlinear:.4f}")
        
        improvement = ((var_mae_linear - var_mae_nonlinear) / var_mae_linear) * 100
        if improvement > 0:
            print(f"    🟢 非线性改进: {improvement:.1f}%")
        else:
            print(f"    🔵 线性更优: {-improvement:.1f}%")
        
        total_mae_linear += var_mae_linear
        total_mae_nonlinear += var_mae_nonlinear
    
    # 总体评估
    overall_mae_linear = total_mae_linear / len([attr_a, attr_b, attr_c])
    overall_mae_nonlinear = total_mae_nonlinear / len([attr_a, attr_b, attr_c])
    overall_improvement = ((overall_mae_linear - overall_mae_nonlinear) / overall_mae_linear) * 100
    
    print(f"\n📈 总体对比:")
    print("-" * 50)
    print(f"线性预测总体MAE: {overall_mae_linear:.4f}")
    print(f"非线性预测总体MAE: {overall_mae_nonlinear:.4f}")
    
    if overall_improvement > 0:
        print(f"🟢 非线性总体改进: {overall_improvement:.1f}%")
        recommendation = "建议使用非线性预测方法"
    else:
        print(f"🔵 线性方法更优: {-overall_improvement:.1f}%")
        recommendation = "建议继续使用线性预测方法"
    
    print(f"\n💡 推荐: {recommendation}")
    
    return {
        'linear_mae': overall_mae_linear,
        'nonlinear_mae': overall_mae_nonlinear,
        'improvement': overall_improvement
    }

if __name__ == "__main__":
    # 可以指定不同的属性进行测试
    result = compare_prediction_methods()
    
    # 也可以测试其他属性组合
    # result = compare_prediction_methods(['natcity', 'natenvir', 'natheal'])
