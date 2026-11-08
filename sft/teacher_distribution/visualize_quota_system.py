"""
配额退火系统可视化工具

提供多种可视化图表来监控和分析系统运行状态
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# 设置样式
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)


class QuotaSystemVisualizer:
    """配额系统可视化器"""
    
    def __init__(self, result_dict=None, result_file=None):
        """
        初始化可视化器
        
        Args:
            result_dict: 系统运行结果字典
            result_file: 结果JSON文件路径
        """
        if result_dict:
            self.result = result_dict
        elif result_file:
            with open(result_file, 'r', encoding='utf-8') as f:
                self.result = json.load(f)
        else:
            raise ValueError("必须提供result_dict或result_file")
        
        self.iteration_history = self.result.get('iteration_history', [])
    
    def plot_convergence(self, save_path=None):
        """
        绘制收敛曲线
        
        展示总偏差和温度随迭代的变化
        """
        if not self.iteration_history:
            print("没有可用的迭代历史数据")
            return
        
        iterations = [h['iteration'] for h in self.iteration_history]
        deviations = [h['total_deviation'] for h in self.iteration_history]
        temperatures = [h['temperature'] for h in self.iteration_history]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # 偏差曲线
        ax1.plot(iterations, deviations, marker='o', linewidth=2.5, 
                markersize=8, label='总偏差', color='#2E86AB')
        ax1.axhline(y=0.02, color='red', linestyle='--', linewidth=2, 
                   label='收敛阈值 (2%)', alpha=0.7)
        ax1.fill_between(iterations, 0, deviations, alpha=0.3, color='#2E86AB')
        ax1.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax1.set_ylabel('总偏差', fontsize=13, fontweight='bold')
        ax1.set_title('配额偏差收敛曲线', fontsize=15, fontweight='bold', pad=15)
        ax1.legend(fontsize=11, loc='upper right')
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(bottom=0)
        
        # 温度曲线
        ax2.plot(iterations, temperatures, marker='s', linewidth=2.5, 
                markersize=8, label='退火温度', color='#FF6B35')
        ax2.fill_between(iterations, 0, temperatures, alpha=0.3, color='#FF6B35')
        ax2.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax2.set_ylabel('温度', fontsize=13, fontweight='bold')
        ax2.set_title('退火温度变化曲线', fontsize=15, fontweight='bold', pad=15)
        ax2.legend(fontsize=11, loc='upper right')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(bottom=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 收敛曲线已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_quota_tracking(self, save_path=None):
        """
        绘制各配额的追踪曲线
        
        展示每个配额的实际值如何逐步接近目标值
        """
        if not self.iteration_history:
            print("没有可用的迭代历史数据")
            return
        
        # 提取配额数据
        quota_names = list(self.iteration_history[0]['quota_satisfaction'].keys())
        
        fig, axes = plt.subplots(len(quota_names), 1, 
                                figsize=(12, 4 * len(quota_names)))
        if len(quota_names) == 1:
            axes = [axes]
        
        for idx, quota_name in enumerate(quota_names):
            iterations = []
            targets = []
            actuals = []
            
            for h in self.iteration_history:
                iterations.append(h['iteration'])
                info = h['quota_satisfaction'][quota_name]
                targets.append(info['target'])
                actuals.append(info['actual'])
            
            ax = axes[idx]
            
            # 绘制目标线和实际线
            ax.plot(iterations, targets, linestyle='--', linewidth=2.5, 
                   label='目标', color='green', alpha=0.8)
            ax.plot(iterations, actuals, marker='o', linewidth=2.5, 
                   markersize=7, label='实际', color='blue')
            
            # 添加容差带
            target_val = targets[0]
            tolerance = 0.05  # 假设5%容差
            ax.fill_between(iterations, 
                          target_val - tolerance, 
                          target_val + tolerance,
                          alpha=0.2, color='green', label='容差范围')
            
            ax.set_xlabel('迭代次数', fontsize=12, fontweight='bold')
            ax.set_ylabel('比例', fontsize=12, fontweight='bold')
            ax.set_title(f'配额: {quota_name}', fontsize=13, fontweight='bold', pad=10)
            ax.legend(fontsize=10, loc='best')
            ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 配额追踪图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_quota_satisfaction(self, save_path=None):
        """
        绘制最终配额满足情况对比
        
        柱状图展示目标vs实际
        """
        final_satisfaction = self.result.get('final_quota_satisfaction', {})
        
        if not final_satisfaction:
            print("没有可用的配额满足数据")
            return
        
        quota_names = list(final_satisfaction.keys())
        targets = [info['target'] for info in final_satisfaction.values()]
        actuals = [info['actual'] for info in final_satisfaction.values()]
        satisfied = [info['satisfied'] for info in final_satisfaction.values()]
        deviations = [info['deviation'] for info in final_satisfaction.values()]
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：目标vs实际对比
        x = np.arange(len(quota_names))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, targets, width, label='目标', 
                       alpha=0.8, color='#4CAF50')
        bars2 = ax1.bar(x + width/2, actuals, width, label='实际',
                       alpha=0.8, color=['#2196F3' if s else '#F44336' for s in satisfied])
        
        ax1.set_xlabel('配额', fontsize=13, fontweight='bold')
        ax1.set_ylabel('比例', fontsize=13, fontweight='bold')
        ax1.set_title('配额目标 vs 实际达成', fontsize=15, fontweight='bold', pad=15)
        ax1.set_xticks(x)
        ax1.set_xticklabels(quota_names, rotation=30, ha='right', fontsize=10)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        # 右图：偏差大小
        colors = ['#4CAF50' if s else '#F44336' for s in satisfied]
        bars = ax2.bar(quota_names, deviations, color=colors, alpha=0.8)
        ax2.axhline(y=0.05, color='orange', linestyle='--', linewidth=2, 
                   label='警戒线 (5%)', alpha=0.7)
        ax2.set_xlabel('配额', fontsize=13, fontweight='bold')
        ax2.set_ylabel('偏差', fontsize=13, fontweight='bold')
        ax2.set_title('配额偏差大小', fontsize=15, fontweight='bold', pad=15)
        ax2.set_xticklabels(quota_names, rotation=30, ha='right', fontsize=10)
        ax2.legend(fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 配额满足图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_weight_evolution(self, save_path=None):
        """
        绘制配额权重演化
        
        展示退火过程中各配额权重如何调整
        """
        if not self.iteration_history:
            print("没有可用的迭代历史数据")
            return
        
        quota_names = list(self.iteration_history[0]['quota_satisfaction'].keys())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for quota_name in quota_names:
            iterations = []
            weights = []
            
            for h in self.iteration_history:
                iterations.append(h['iteration'])
                weights.append(h['quota_satisfaction'][quota_name]['weight'])
            
            ax.plot(iterations, weights, marker='o', linewidth=2.5, 
                   markersize=7, label=quota_name, alpha=0.8)
        
        ax.set_xlabel('迭代次数', fontsize=13, fontweight='bold')
        ax.set_ylabel('权重', fontsize=13, fontweight='bold')
        ax.set_title('配额权重演化（退火过程）', fontsize=15, fontweight='bold', pad=15)
        ax.legend(fontsize=10, loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.axhline(y=1.0, color='gray', linestyle=':', linewidth=1.5, 
                  label='初始权重', alpha=0.6)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 权重演化图已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_dashboard(self, save_path=None):
        """
        绘制综合仪表盘
        
        在一张图中展示多个关键指标
        """
        fig = plt.figure(figsize=(16, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 1. 收敛曲线
        ax1 = fig.add_subplot(gs[0, :2])
        iterations = [h['iteration'] for h in self.iteration_history]
        deviations = [h['total_deviation'] for h in self.iteration_history]
        ax1.plot(iterations, deviations, marker='o', linewidth=2, markersize=6, color='#2E86AB')
        ax1.axhline(y=0.02, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        ax1.fill_between(iterations, 0, deviations, alpha=0.2, color='#2E86AB')
        ax1.set_title('收敛曲线', fontsize=12, fontweight='bold')
        ax1.set_xlabel('迭代', fontsize=10)
        ax1.set_ylabel('总偏差', fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # 2. 关键指标卡片
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        metrics_text = f"""
        总迭代: {self.result['total_iterations']}
        
        最佳得分: {self.result['best_score']:.4f}
        
        收敛状态: {'✓ 已收敛' if self.result['convergence_achieved'] else '✗ 未收敛'}
        
        样本数: {self.result['best_samples_count']}
        """
        ax2.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        # 3. 配额满足情况
        ax3 = fig.add_subplot(gs[1, :])
        final_satisfaction = self.result.get('final_quota_satisfaction', {})
        quota_names = list(final_satisfaction.keys())
        targets = [info['target'] for info in final_satisfaction.values()]
        actuals = [info['actual'] for info in final_satisfaction.values()]
        satisfied = [info['satisfied'] for info in final_satisfaction.values()]
        
        x = np.arange(len(quota_names))
        width = 0.35
        ax3.bar(x - width/2, targets, width, label='目标', alpha=0.7, color='green')
        ax3.bar(x + width/2, actuals, width, label='实际', alpha=0.7,
               color=['blue' if s else 'red' for s in satisfied])
        ax3.set_title('配额满足情况', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(quota_names, rotation=25, ha='right', fontsize=9)
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # 4. 温度变化
        ax4 = fig.add_subplot(gs[2, 0])
        temperatures = [h['temperature'] for h in self.iteration_history]
        ax4.plot(iterations, temperatures, marker='s', linewidth=2, 
                markersize=6, color='#FF6B35')
        ax4.fill_between(iterations, 0, temperatures, alpha=0.2, color='#FF6B35')
        ax4.set_title('退火温度', fontsize=12, fontweight='bold')
        ax4.set_xlabel('迭代', fontsize=10)
        ax4.set_ylabel('温度', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # 5. 偏差分布
        ax5 = fig.add_subplot(gs[2, 1])
        deviations_final = [info['deviation'] for info in final_satisfaction.values()]
        ax5.bar(range(len(deviations_final)), deviations_final, 
               color=['green' if d < 0.05 else 'orange' if d < 0.1 else 'red' 
                      for d in deviations_final], alpha=0.7)
        ax5.axhline(y=0.05, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
        ax5.set_title('最终偏差分布', fontsize=12, fontweight='bold')
        ax5.set_xlabel('配额索引', fontsize=10)
        ax5.set_ylabel('偏差', fontsize=10)
        ax5.grid(True, alpha=0.3, axis='y')
        
        # 6. 满足率饼图
        ax6 = fig.add_subplot(gs[2, 2])
        n_satisfied = sum(satisfied)
        n_total = len(satisfied)
        sizes = [n_satisfied, n_total - n_satisfied]
        colors = ['#4CAF50', '#F44336']
        labels = [f'满足 ({n_satisfied})', f'未满足 ({n_total - n_satisfied})']
        ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 10})
        ax6.set_title('配额满足率', fontsize=12, fontweight='bold')
        
        # 总标题
        fig.suptitle('配额退火系统 - 综合仪表盘', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ 仪表盘已保存: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def generate_all_plots(self, output_dir='./visualization_output'):
        """
        生成所有可视化图表
        
        Args:
            output_dir: 输出目录
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True, parents=True)
        
        print(f"\n📊 开始生成可视化图表...")
        print(f"输出目录: {output_path}")
        
        # 生成各种图表
        self.plot_convergence(output_path / 'convergence.png')
        self.plot_quota_tracking(output_path / 'quota_tracking.png')
        self.plot_quota_satisfaction(output_path / 'quota_satisfaction.png')
        self.plot_weight_evolution(output_path / 'weight_evolution.png')
        self.plot_dashboard(output_path / 'dashboard.png')
        
        print(f"\n✅ 所有图表已生成完毕!")
        print(f"请查看目录: {output_path}")


def visualize_from_file(json_file, output_dir='./visualization_output'):
    """
    从JSON结果文件生成可视化
    
    Args:
        json_file: 结果JSON文件路径
        output_dir: 输出目录
    """
    try:
        visualizer = QuotaSystemVisualizer(result_file=json_file)
        visualizer.generate_all_plots(output_dir)
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


def visualize_from_dict(result_dict, output_dir='./visualization_output'):
    """
    从结果字典生成可视化
    
    Args:
        result_dict: 系统运行结果字典
        output_dir: 输出目录
    """
    try:
        visualizer = QuotaSystemVisualizer(result_dict=result_dict)
        visualizer.generate_all_plots(output_dir)
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='配额系统可视化工具')
    parser.add_argument('--input', type=str, required=True,
                       help='输入的JSON结果文件路径')
    parser.add_argument('--output', type=str, default='./visualization_output',
                       help='输出目录（默认: ./visualization_output）')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("配额退火系统 - 可视化工具")
    print("=" * 60)
    
    visualize_from_file(args.input, args.output)

