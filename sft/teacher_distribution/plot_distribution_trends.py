import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from get_distribution import (load_and_process_data, year_stats, ATTR_NAMES, levels, 
                             process_data_with_attrs, marginal_from_Q, normalize)
import matplotlib.font_manager as fm

# 设置中文字体
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
# plt.rcParams['axes.unicode_minus'] = False

def extract_real_2024_data():
    """
    提取2024年真实GSS数据的概率分布
    
    Returns:
        dict: 包含三个变量在2024年真实数据中的概率分布
    """
    # 读取2024年真实数据
    df_2024 = pd.read_csv('/home/cyyuan/FAS_PRO/data/gss/GSS2024.csv')
    
    # 确保属性列存在
    attr_a, attr_b, attr_c = ATTR_NAMES
    missing_cols = [col for col in ATTR_NAMES if col not in df_2024.columns]
    if missing_cols:
        raise ValueError(f"2024年数据文件缺少列：{missing_cols}")
    
    print(f"2024年真实数据总样本量：{len(df_2024)}")
    
    # 数据预处理：正确处理缺失值和无效值
    df_2024_cleaned = df_2024.copy()
    
    # 将缺失值和空字符串替换为NaN
    for col in ATTR_NAMES:
        df_2024_cleaned[col] = pd.to_numeric(df_2024_cleaned[col], errors='coerce')
    
    # 只保留所有三个变量都有有效值（1,2,3）的行
    valid_mask = pd.Series(True, index=df_2024_cleaned.index)
    for col in ATTR_NAMES:
        valid_mask &= df_2024_cleaned[col].isin([1, 2, 3])
    
    df_2024_filtered = df_2024_cleaned.loc[valid_mask].copy()
    
    print(f"同时具有三个变量有效值的样本量：{len(df_2024_filtered)}")
    
    # 检查每个变量的有效数据量
    for col in ATTR_NAMES:
        valid_count = df_2024_cleaned[col].isin([1, 2, 3]).sum()
        print(f"{col} 有效数据量: {valid_count}")
    
    # 计算各变量的概率分布
    real_2024_stats = {}
    
    for var_name in [attr_a, attr_b, attr_c]:
        # 转换为字符串格式以匹配历史数据格式
        df_2024_filtered[var_name] = df_2024_filtered[var_name].astype(int).astype(str)
        
        # 计算频次分布
        counts = df_2024_filtered[var_name].value_counts().reindex(levels, fill_value=0).astype(float)
        # 归一化为概率分布
        probs = normalize(counts)
        real_2024_stats[f"P_{var_name}"] = probs
        
        print(f"\n{var_name} 2024年真实分布:")
        for level in levels:
            count = counts[level]
            prob = probs[level]
            print(f"  Level {level}: {count:4.0f} 个样本 ({prob:.4f})")
    
    return real_2024_stats

def plot_distribution_trends():
    """
    绘制每年变量概率分布的变化趋势，包含2024年预测值
    """
    # 加载历史数据
    print("正在加载历史数据...")
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    
    # 获取2024年真实数据
    print("正在加载2024年真实数据...")
    real_2024_stats = extract_real_2024_data()
    
    # 准备数据
    attr_a, attr_b, attr_c = ATTR_NAMES
    
    # 获取2024年预测数据
    print("正在计算2024年预测值...")
    Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints = process_data_with_attrs()
    pred_2024_stats = {
        f"P_{attr_a}": Pa_pred,
        f"P_{attr_b}": Pb_pred,
        f"P_{attr_c}": Pc_pred
    }
    
    # 提取年份
    years = sorted(stats_by_year.keys())
    print(f"历史年份: {years}")
    print("对比年份: 2024 (预测 vs 真实)")
    
    # 更新年份列表
    all_years = sorted(stats_by_year.keys())
    historical_years = years
    comparison_year = 2024
    
    # 创建图形
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    fig.suptitle('Variables Probability Distribution Trends (Historical + 2024 Prediction vs Real)', fontsize=16, fontweight='bold')
    
    # 定义颜色
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # 蓝色、橙色、绿色（历史数据）
    pred_colors = ['#FF69B4', '#FF8C00', '#32CD32']  # 粉红、深橙、青绿（2024预测数据）
    real_colors = ['#8B0000', '#B22222', '#006400']  # 深红、火砖红、深绿（2024真实数据）
    level_names = ['Level 1', 'Level 2', 'Level 3']
    
    # 为每个变量和每个级别绘制趋势线
    for var_idx, var_name in enumerate([attr_a, attr_b, attr_c]):
        for level_idx, level in enumerate(levels):
            ax = axes[var_idx, level_idx]
            
            # 提取该变量该级别在历史年份的概率
            historical_probs = []
            for year in historical_years:
                prob = stats_by_year[year][f"P_{var_name}"][level]
                historical_probs.append(prob)
            
            # 获取2024年预测值和真实数据值
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            
            # 绘制历史趋势线
            ax.plot(historical_years, historical_probs, marker='o', linewidth=2, markersize=6, 
                   color=colors[level_idx], label=f'{level_names[level_idx]} (Historical)', alpha=0.8)
            ax.fill_between(historical_years, historical_probs, alpha=0.2, color=colors[level_idx])
            
            # 绘制2024年预测点和连接线
            ax.plot([historical_years[-1], comparison_year], [historical_probs[-1], pred_prob], 
                   '--', color=pred_colors[level_idx], linewidth=2, alpha=0.8)
            ax.plot(comparison_year, pred_prob, marker='^', markersize=10, 
                   color=pred_colors[level_idx], label=f'{level_names[level_idx]} (Predicted 2024)', 
                   markeredgecolor='black', markeredgewidth=1)
            
            # 绘制2024年真实数据点
            ax.plot(comparison_year + 0.1, real_prob, marker='s', markersize=10, 
                   color=real_colors[level_idx], label=f'{level_names[level_idx]} (Real 2024)', 
                   markeredgecolor='black', markeredgewidth=1)
            
            # 设置标题和标签
            ax.set_title(f'{var_name} - {level_names[level_idx]}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Year')
            ax.set_ylabel('Probability')
            ax.grid(True, alpha=0.3)
            
            # 设置y轴范围
            all_probs = historical_probs + [pred_prob, real_prob]
            ax.set_ylim(0, max(all_probs) * 1.15)
            
            # 添加数值标签（历史数据）
            for i, (year, prob) in enumerate(zip(historical_years, historical_probs)):
                ax.annotate(f'{prob:.3f}', (year, prob), 
                           textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
            
            # 添加2024年预测值标签
            ax.annotate(f'P:{pred_prob:.3f}', (comparison_year, pred_prob), 
                       textcoords="offset points", xytext=(-15,10), ha='center', fontsize=8, 
                       fontweight='bold', color=pred_colors[level_idx])
            
            # 添加2024年真实数据标签
            ax.annotate(f'R:{real_prob:.3f}', (comparison_year + 0.1, real_prob), 
                       textcoords="offset points", xytext=(15,10), ha='center', fontsize=8, 
                       fontweight='bold', color=real_colors[level_idx])
            
            # 计算并显示差异，根据误差大小使用不同颜色
            diff = real_prob - pred_prob
            abs_diff = abs(diff)
            
            # 根据误差大小设置颜色
            if abs_diff > 0.05:  # 大误差 > 5%
                diff_color = 'red'
                fontweight = 'bold'
                fontsize = 8
            elif abs_diff > 0.02:  # 中等误差 2-5%
                diff_color = 'orange'
                fontweight = 'bold'
                fontsize = 7
            else:  # 小误差 < 2%
                diff_color = 'green'
                fontweight = 'normal'
                fontsize = 7
            
            ax.annotate(f'Δ:{diff:+.3f}', (comparison_year + 0.05, (pred_prob + real_prob) / 2), 
                       textcoords="offset points", xytext=(0,-10), ha='center', fontsize=fontsize, 
                       fontweight=fontweight, color=diff_color)
            
            # 添加图例（只在第一个子图添加）
            if var_idx == 0 and level_idx == 0:
                ax.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'/home/cyyuan/FAS_PRO/sft/graphs/distribution_trends_EDC.png')
    #plt.show()
    
    # 创建汇总趋势图
    fig2, axes2 = plt.subplots(1, 3, figsize=(20, 6))
    fig2.suptitle('Three Variables Probability Distribution Trends Comparison (Historical + 2024 Prediction vs Real)', fontsize=16, fontweight='bold')
    
    for var_idx, var_name in enumerate([attr_a, attr_b, attr_c]):
        ax = axes2[var_idx]
        
        # 为每个级别绘制趋势线
        for level_idx, level in enumerate(levels):
            # 历史数据
            historical_probs = []
            for year in historical_years:
                prob = stats_by_year[year][f"P_{var_name}"][level]
                historical_probs.append(prob)
            
            # 2024年预测值和真实数据值
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            
            # 绘制历史趋势线
            ax.plot(historical_years, historical_probs, marker='o', linewidth=2, markersize=6, 
                   color=colors[level_idx], label=f'{level_names[level_idx]} (Historical)', alpha=0.8)
            
            # 绘制2024年预测点和连接线
            ax.plot([historical_years[-1], comparison_year], [historical_probs[-1], pred_prob], 
                   '--', color=pred_colors[level_idx], linewidth=2, alpha=0.8)
            ax.plot(comparison_year, pred_prob, marker='^', markersize=10, 
                   color=pred_colors[level_idx], label=f'{level_names[level_idx]} (2024 Pred)', 
                   markeredgecolor='black', markeredgewidth=1)
            
            # 绘制2024年真实数据点
            ax.plot(comparison_year + 0.1, real_prob, marker='s', markersize=10, 
                   color=real_colors[level_idx], label=f'{level_names[level_idx]} (2024 Real)', 
                   markeredgecolor='black', markeredgewidth=1)
        
        ax.set_title(f'{var_name} Levels Probability Changes', fontsize=14, fontweight='bold')
        ax.set_xlabel('Year')
        ax.set_ylabel('Probability')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'/home/cyyuan/FAS_PRO/sft/graphs/distribution_trends_comparison_EDC.png')
    #plt.show()
    
    # 创建数据表格
    print("\n=== Detailed Data Table (Historical + 2024 Prediction vs Real) ===")
    for var_name in [attr_a, attr_b, attr_c]:
        print(f"\n{var_name} Probability Distribution by Year:")
        data_for_table = []
        
        # 添加历史数据
        for year in all_years:
            year_data = [year]
            for level in levels:
                prob = stats_by_year[year][f"P_{var_name}"][level]
                year_data.append(prob)
            year_data.append("(Historical)")
            data_for_table.append(year_data)
        
        # 添加2024年预测数据
        pred_data = ["2024 (Pred)"]
        for level in levels:
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            pred_data.append(pred_prob)
        pred_data.append("(Predicted)")
        data_for_table.append(pred_data)
        
        # 添加2024年真实数据
        real_data = ["2024 (Real)"]
        for level in levels:
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            real_data.append(real_prob)
        real_data.append("(Real Data)")
        data_for_table.append(real_data)
        
        # 添加差异行
        diff_data = ["2024 (Diff)"]
        for level in levels:
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            diff = real_prob - pred_prob
            diff_data.append(diff)
        diff_data.append("(Real - Pred)")
        data_for_table.append(diff_data)
        
        df_table = pd.DataFrame(data_for_table, columns=['Year'] + [f'Level {l}' for l in levels] + ['Type'])
        print(df_table.round(4))
    
    # 绘制热力图
    fig3, axes3 = plt.subplots(1, 3, figsize=(20, 8))
    fig3.suptitle('Probability Distribution Heatmap (Historical + 2024 Prediction vs Real)', fontsize=16, fontweight='bold')
    
    for var_idx, var_name in enumerate([attr_a, attr_b, attr_c]):
        # 准备热力图数据（包含历史、预测和真实数据）
        heatmap_data = []
        year_labels = []
        
        # 添加历史数据
        for year in all_years:
            year_probs = []
            for level in levels:
                prob = stats_by_year[year][f"P_{var_name}"][level]
                year_probs.append(prob)
            heatmap_data.append(year_probs)
            year_labels.append(str(year))
        
        # 添加2024年预测数据
        pred_probs = []
        for level in levels:
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            pred_probs.append(pred_prob)
        heatmap_data.append(pred_probs)
        year_labels.append('2024P')
        
        # 添加2024年真实数据
        real_probs = []
        for level in levels:
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            real_probs.append(real_prob)
        heatmap_data.append(real_probs)
        year_labels.append('2024R')
        
        heatmap_df = pd.DataFrame(heatmap_data, 
                                 index=year_labels, 
                                 columns=[f'Level {l}' for l in levels])
        
        # 创建自定义颜色映射，突出显示预测值
        cmap = sns.color_palette("YlOrRd", as_cmap=True)
        
        sns.heatmap(heatmap_df, annot=True, fmt='.3f', cmap=cmap, 
                   ax=axes3[var_idx], cbar_kws={'label': 'Probability'},
                   linewidths=0.5, linecolor='white')
        
        axes3[var_idx].set_title(f'{var_name} Probability Distribution Heatmap', fontweight='bold')
        axes3[var_idx].set_xlabel('Level')
        axes3[var_idx].set_ylabel('Year (P=Predicted, R=Real)')
        
        # 突出显示预测和真实数据行
        pred_row_idx = year_labels.index('2024P')
        real_row_idx = year_labels.index('2024R')
        
        # 预测行用蓝色边框
        for j in range(len(levels)):
            rect = plt.Rectangle((j, pred_row_idx), 1, 1, fill=False, 
                               edgecolor='blue', linewidth=2)
            axes3[var_idx].add_patch(rect)
        
        # 真实数据行用红色边框
        for j in range(len(levels)):
            rect = plt.Rectangle((j, real_row_idx), 1, 1, fill=False, 
                               edgecolor='red', linewidth=2)
            axes3[var_idx].add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(f'/home/cyyuan/FAS_PRO/sft/graphs/distribution_heatmap_EDC.png')
    #plt.show()

def plot_trend_analysis():
    """
    更详细的趋势分析（包含2024年真实数据）
    """
    # 加载历史数据
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    historical_years = sorted(stats_by_year.keys())
    
    # 准备数据
    attr_a, attr_b, attr_c = ATTR_NAMES
    
    # 获取2024年真实数据和预测数据
    real_2024_stats = extract_real_2024_data()
    Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints = process_data_with_attrs()
    pred_2024_stats = {
        f"P_{attr_a}": Pa_pred,
        f"P_{attr_b}": Pb_pred,
        f"P_{attr_c}": Pc_pred
    }
    
    all_years = sorted(stats_by_year.keys())
    
    # 计算趋势指标
    print("\n=== Trend Analysis (Historical + 2024 Prediction vs Real) ===")
    
    for var_name in [attr_a, attr_b, attr_c]:
        print(f"\n{var_name} Trend Analysis:")
        for level in levels:
            # 历史数据
            historical_probs = []
            for year in historical_years:
                prob = stats_by_year[year][f"P_{var_name}"][level]
                historical_probs.append(prob)
            
            # 2024年预测值和真实数据值
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            
            # 计算历史线性趋势
            x = np.array(historical_years)
            y = np.array(historical_probs)
            z = np.polyfit(x, y, 1)
            slope = z[0]
            
            # 使用趋势预测2024年的值
            predicted_by_trend = slope * 2024 + z[1]
            predicted_by_trend = max(0.0, min(1.0, predicted_by_trend))  # 限制在[0,1]范围内
            
            # 计算变化率
            if len(historical_probs) > 1:
                historical_change = historical_probs[-1] - historical_probs[0]
                avg_annual_change = historical_change / (historical_years[-1] - historical_years[0])
                
                # 预测与真实数据的差异
                pred_vs_real = real_prob - pred_prob
                real_vs_trend = real_prob - predicted_by_trend
                pred_vs_trend = pred_prob - predicted_by_trend
                
                print(f"  Level {level}:")
                print(f"    Historical Start ({historical_years[0]}): {historical_probs[0]:.4f}")
                print(f"    Historical End ({historical_years[-1]}): {historical_probs[-1]:.4f}")
                print(f"    Historical Change: {historical_change:+.4f}")
                print(f"    Avg Annual Change: {avg_annual_change:+.6f}")
                print(f"    Linear Trend Slope: {slope:+.6f}")
                print(f"    Trend-based 2024 Prediction: {predicted_by_trend:.4f}")
                print(f"    Model-based 2024 Prediction: {pred_prob:.4f}")
                print(f"    Real 2024 Data: {real_prob:.4f}")
                print(f"    Model vs Trend Difference: {pred_vs_trend:+.4f}")
                print(f"    Real vs Model Difference: {pred_vs_real:+.4f}")
                print(f"    Real vs Trend Difference: {real_vs_trend:+.4f}")
                print(f"    2022→2024 Real Change: {real_prob - historical_probs[-1]:+.4f}")
                print(f"    2022→2024 Pred Change: {pred_prob - historical_probs[-1]:+.4f}")

def evaluate_prediction_performance():
    """
    评估预测效果并生成报告
    """
    # 获取数据
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    real_2024_stats = extract_real_2024_data()
    Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints = process_data_with_attrs()
    
    attr_a, attr_b, attr_c = ATTR_NAMES
    pred_2024_stats = {
        f"P_{attr_a}": Pa_pred,
        f"P_{attr_b}": Pb_pred,
        f"P_{attr_c}": Pc_pred
    }
    
    print("\n" + "="*60)
    print("🎯 预测效果评估报告")
    print("="*60)
    
    total_mae = 0
    total_rmse = 0
    variable_count = 0
    
    for var_name in [attr_a, attr_b, attr_c]:
        print(f"\n📊 {var_name.upper()} 预测效果:")
        print("-" * 40)
        
        var_mae = 0
        var_rmse = 0
        
        for level in levels:
            pred_prob = pred_2024_stats[f"P_{var_name}"][level]
            real_prob = real_2024_stats[f"P_{var_name}"][level]
            
            diff = real_prob - pred_prob
            abs_diff = abs(diff)
            squared_diff = diff ** 2
            
            var_mae += abs_diff
            var_rmse += squared_diff
            
            # 效果评级
            if abs_diff < 0.01:
                grade = "🟢 优秀"
            elif abs_diff < 0.02:
                grade = "🟡 良好"
            elif abs_diff < 0.05:
                grade = "🟠 一般"
            else:
                grade = "🔴 较差"
            
            print(f"  Level {level}: 预测{pred_prob:.3f} | 真实{real_prob:.3f} | 误差{diff:+.3f} | {grade}")
        
        var_mae /= len(levels)
        var_rmse = (var_rmse / len(levels)) ** 0.5
        
        print(f"  平均绝对误差 (MAE): {var_mae:.4f}")
        print(f"  均方根误差 (RMSE): {var_rmse:.4f}")
        
        total_mae += var_mae
        total_rmse += var_rmse
        variable_count += 1
    
    # 总体评估
    overall_mae = total_mae / variable_count
    overall_rmse = total_rmse / variable_count
    
    print(f"\n📈 总体预测效果:")
    print("-" * 40)
    print(f"平均MAE: {overall_mae:.4f}")
    print(f"平均RMSE: {overall_rmse:.4f}")
    
    if overall_mae < 0.02:
        overall_grade = "🟢 优秀"
    elif overall_mae < 0.03:
        overall_grade = "🟡 良好"
    elif overall_mae < 0.05:
        overall_grade = "🟠 一般"
    else:
        overall_grade = "🔴 需要改进"
    
    print(f"总体评级: {overall_grade}")
    
    # 改进建议
    print(f"\n💡 改进建议:")
    print("-" * 40)
    if overall_mae > 0.03:
        print("• 考虑使用更复杂的非线性模型")
        print("• 增加更多历史数据点")
        print("• 考虑添加外部影响因素")
    else:
        print("• 预测效果良好，可以考虑应用到实际场景")
    
    return overall_mae, overall_rmse

if __name__ == "__main__":
    # 绘制分布趋势图
    plot_distribution_trends()
    
    # 进行趋势分析
    plot_trend_analysis()
    
    # 评估预测效果
    evaluate_prediction_performance()
