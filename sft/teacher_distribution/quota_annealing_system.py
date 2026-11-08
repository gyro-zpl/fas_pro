"""
配额退火和子群体判别系统
用于社会模拟微调数据集的迭代优化生成

功能：
1. 配额退火：根据目标分布配额动态调整采样权重
2. 子群体判别：评估生成样本的质量和合理性
3. 迭代修正：循环优化直到达到目标分布
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# 导入现有的分布生成工具
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入函数和常量
from make_finetune_datasets import (
    ipf_with_tables, sample_from_joint, normalize_series,
    ATTR_A, ATTR_B, ATTR_C, ATTR_CONFIG, levels,
    initialize_distributions_from_real_data
)

# 全局变量，将在系统初始化时赋值
P_A_2024 = None
P_B_2024 = None
P_C_2024 = None
P_BC_2024 = None
P_AB_2024 = None
P_AC_2024 = None


# =========================
# 1. 配额定义和管理
# =========================

@dataclass
class Quota:
    """
    配额定义类
    
    配额可以是单维度或多维度的目标分布约束
    """
    name: str  # 配额名称
    dimensions: Dict[str, str]  # 维度和值的组合，例如 {'age': '50', 'gender': 'male'}
    target_proportion: float  # 目标比例
    tolerance: float = 0.05  # 容差范围
    weight: float = 1.0  # 配额权重（可通过退火调整）
    initial_weight: float = 1.0  # 初始权重
    priority: int = 1  # 优先级（1-5，5最高）
    
    def __post_init__(self):
        self.initial_weight = self.weight
    
    def is_satisfied(self, actual_proportion: float) -> bool:
        """判断配额是否满足"""
        return abs(actual_proportion - self.target_proportion) <= self.tolerance
    
    def deviation(self, actual_proportion: float) -> float:
        """计算偏离程度"""
        return abs(actual_proportion - self.target_proportion)
    
    def relative_deviation(self, actual_proportion: float) -> float:
        """计算相对偏离程度"""
        if self.target_proportion == 0:
            return abs(actual_proportion)
        return abs(actual_proportion - self.target_proportion) / max(self.target_proportion, 0.01)
    
    def __repr__(self):
        dims = ', '.join([f"{k}={v}" for k, v in self.dimensions.items()])
        return f"Quota({dims}: target={self.target_proportion:.2%}, weight={self.weight:.2f})"


@dataclass
class QuotaManager:
    """配额管理器"""
    quotas: List[Quota] = field(default_factory=list)
    
    def add_quota(self, quota: Quota):
        """添加配额"""
        self.quotas.append(quota)
    
    def add_single_dim_quota(self, attr: str, value: str, target_prop: float, 
                            tolerance: float = 0.05, weight: float = 1.0, priority: int = 1):
        """添加单维度配额"""
        quota = Quota(
            name=f"{attr}={value}",
            dimensions={attr: value},
            target_proportion=target_prop,
            tolerance=tolerance,
            weight=weight,
            priority=priority
        )
        self.add_quota(quota)
    
    def add_multi_dim_quota(self, dimensions: Dict[str, str], target_prop: float,
                           tolerance: float = 0.05, weight: float = 1.0, priority: int = 1):
        """添加多维度配额"""
        name = '&'.join([f"{k}={v}" for k, v in dimensions.items()])
        quota = Quota(
            name=name,
            dimensions=dimensions,
            target_proportion=target_prop,
            tolerance=tolerance,
            weight=weight,
            priority=priority
        )
        self.add_quota(quota)
    
    def get_quota_by_priority(self) -> List[Quota]:
        """按优先级排序返回配额"""
        return sorted(self.quotas, key=lambda q: q.priority, reverse=True)
    
    def reset_weights(self):
        """重置所有配额权重到初始值"""
        for quota in self.quotas:
            quota.weight = quota.initial_weight


# =========================
# 2. 配额退火机制
# =========================

class QuotaAnnealing:
    """
    配额退火系统
    
    根据配额的可靠性和历史表现动态调整权重
    """
    
    def __init__(self, initial_temperature: float = 1.0, 
                 cooling_rate: float = 0.95,
                 min_temperature: float = 0.1):
        self.temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.quota_history = defaultdict(list)  # 记录每个配额的历史表现
    
    def anneal_quota_weights(self, quota_manager: QuotaManager, 
                            actual_proportions: Dict[str, float],
                            iteration: int) -> Dict[str, float]:
        """
        根据实际表现退火配额权重
        
        Args:
            quota_manager: 配额管理器
            actual_proportions: 实际达到的比例
            iteration: 当前迭代次数
            
        Returns:
            调整后的权重字典
        """
        adjusted_weights = {}
        
        for quota in quota_manager.quotas:
            actual_prop = actual_proportions.get(quota.name, 0.0)
            deviation = quota.relative_deviation(actual_prop)
            
            # 记录历史表现
            self.quota_history[quota.name].append({
                'iteration': iteration,
                'deviation': deviation,
                'actual': actual_prop,
                'target': quota.target_proportion
            })
            
            # 计算历史趋势：是否在改善
            is_improving = self._is_improving(quota.name)
            
            # 计算可靠性得分 (0-1)
            reliability_score = self._calculate_reliability(quota.name, deviation)
            
            # 退火调整权重
            if is_improving:
                # 改善中：保持或略微增加权重
                adjustment = 1.0 + 0.1 * self.temperature * quota.priority / 5
            else:
                # 未改善：根据可靠性降低权重
                adjustment = reliability_score * (0.5 + 0.5 * self.temperature)
            
            # 应用温度调整
            new_weight = quota.initial_weight * adjustment
            new_weight = max(0.1, min(2.0, new_weight))  # 限制在合理范围
            
            quota.weight = new_weight
            adjusted_weights[quota.name] = new_weight
        
        # 降温
        self.temperature = max(self.min_temperature, self.temperature * self.cooling_rate)
        
        return adjusted_weights
    
    def _is_improving(self, quota_name: str, window: int = 3) -> bool:
        """判断配额是否在改善"""
        history = self.quota_history[quota_name]
        if len(history) < 2:
            return True  # 初期默认改善
        
        recent = history[-min(window, len(history)):]
        deviations = [h['deviation'] for h in recent]
        
        # 简单趋势判断：最近的偏差是否在减小
        if len(deviations) >= 2:
            return deviations[-1] < deviations[0]
        return True
    
    def _calculate_reliability(self, quota_name: str, current_deviation: float) -> float:
        """
        计算配额的可靠性得分
        
        可靠性基于：
        1. 当前偏差大小
        2. 历史稳定性
        3. 达成率
        """
        history = self.quota_history[quota_name]
        
        # 偏差得分：偏差越小得分越高
        deviation_score = np.exp(-5 * current_deviation)
        
        # 稳定性得分：历史偏差的方差越小越稳定
        if len(history) >= 3:
            recent_deviations = [h['deviation'] for h in history[-5:]]
            stability = 1.0 / (1.0 + np.std(recent_deviations))
        else:
            stability = 0.5
        
        # 综合得分
        reliability = 0.7 * deviation_score + 0.3 * stability
        return np.clip(reliability, 0.0, 1.0)
    
    def get_annealing_report(self) -> Dict:
        """生成退火报告"""
        report = {
            'current_temperature': self.temperature,
            'quotas_performance': {}
        }
        
        for quota_name, history in self.quota_history.items():
            if history:
                latest = history[-1]
                report['quotas_performance'][quota_name] = {
                    'iterations': len(history),
                    'latest_deviation': latest['deviation'],
                    'is_improving': self._is_improving(quota_name),
                    'reliability': self._calculate_reliability(quota_name, latest['deviation'])
                }
        
        return report


# =========================
# 3. 子群体判别器
# =========================

class SubgroupDiscriminator:
    """
    子群体判别器
    
    评估生成样本在特定子群体内的质量和合理性
    """
    
    def __init__(self, attr_names: List[str], levels: List[str]):
        self.attr_names = attr_names
        self.levels = levels
        self.quality_thresholds = {
            'diversity': 0.3,  # 多样性阈值
            'concentration': 0.8,  # 集中度阈值
            'balance': 0.2  # 平衡性阈值
        }
    
    def evaluate_subgroup(self, samples: List[List[str]], 
                         subgroup_condition: Dict[str, str]) -> Dict:
        """
        评估特定子群体的质量
        
        Args:
            samples: 生成的样本 [[attr_a_val, attr_b_val, attr_c_val], ...]
            subgroup_condition: 子群体条件，例如 {'age': '50', 'gender': 'male'}
            
        Returns:
            质量评估报告
        """
        # 筛选符合条件的子群体样本
        subgroup_samples = self._filter_subgroup(samples, subgroup_condition)
        
        if len(subgroup_samples) == 0:
            return {
                'size': 0,
                'proportion': 0.0,
                'quality_score': 0.0,
                'warning': 'Empty subgroup'
            }
        
        # 计算子群体指标
        diversity = self._calculate_diversity(subgroup_samples)
        concentration = self._calculate_concentration(subgroup_samples)
        balance = self._calculate_balance(subgroup_samples)
        consistency = self._calculate_consistency(subgroup_samples)
        
        # 综合质量得分
        quality_score = (
            0.3 * diversity +
            0.2 * (1.0 - concentration) +  # 过度集中是不好的
            0.3 * balance +
            0.2 * consistency
        )
        
        # 识别潜在问题
        issues = self._identify_issues(diversity, concentration, balance)
        
        return {
            'size': len(subgroup_samples),
            'proportion': len(subgroup_samples) / len(samples),
            'diversity': diversity,
            'concentration': concentration,
            'balance': balance,
            'consistency': consistency,
            'quality_score': quality_score,
            'issues': issues,
            'passes_quality': quality_score >= 0.6
        }
    
    def _filter_subgroup(self, samples: List[List[str]], 
                        condition: Dict[str, str]) -> List[List[str]]:
        """筛选符合条件的子群体"""
        filtered = []
        attr_indices = {attr: i for i, attr in enumerate(self.attr_names)}
        
        for sample in samples:
            match = True
            for attr, value in condition.items():
                if attr in attr_indices:
                    idx = attr_indices[attr]
                    if sample[idx] != value:
                        match = False
                        break
            if match:
                filtered.append(sample)
        
        return filtered
    
    def _calculate_diversity(self, subgroup_samples: List[List[str]]) -> float:
        """
        计算子群体内的多样性（熵）
        
        多样性高表示子群体内部各属性分布较均匀
        """
        if len(subgroup_samples) == 0:
            return 0.0
        
        # 计算每个非条件属性的熵
        entropies = []
        for i, attr in enumerate(self.attr_names):
            values = [sample[i] for sample in subgroup_samples]
            entropy = self._calculate_entropy(values)
            entropies.append(entropy)
        
        # 归一化熵（最大熵 = log(n_levels)）
        max_entropy = np.log(len(self.levels))
        normalized_entropy = np.mean(entropies) / max_entropy if max_entropy > 0 else 0
        
        return normalized_entropy
    
    def _calculate_entropy(self, values: List[str]) -> float:
        """计算熵"""
        unique, counts = np.unique(values, return_counts=True)
        probs = counts / len(values)
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        return entropy
    
    def _calculate_concentration(self, subgroup_samples: List[List[str]]) -> float:
        """
        计算集中度
        
        集中度高表示某些值过度集中
        """
        if len(subgroup_samples) == 0:
            return 0.0
        
        concentrations = []
        for i in range(len(self.attr_names)):
            values = [sample[i] for sample in subgroup_samples]
            unique, counts = np.unique(values, return_counts=True)
            max_freq = np.max(counts) / len(values)
            concentrations.append(max_freq)
        
        return np.mean(concentrations)
    
    def _calculate_balance(self, subgroup_samples: List[List[str]]) -> float:
        """
        计算平衡性
        
        平衡性基于各属性分布与期望分布的KL散度
        """
        if len(subgroup_samples) == 0:
            return 0.0
        
        balances = []
        uniform_dist = np.ones(len(self.levels)) / len(self.levels)
        
        for i in range(len(self.attr_names)):
            values = [sample[i] for sample in subgroup_samples]
            observed_dist = np.zeros(len(self.levels))
            for level_idx, level in enumerate(self.levels):
                observed_dist[level_idx] = values.count(level) / len(values)
            
            # 计算与均匀分布的KL散度
            kl_div = np.sum(observed_dist * np.log((observed_dist + 1e-10) / (uniform_dist + 1e-10)))
            balance_score = np.exp(-kl_div)  # 转换为0-1分数
            balances.append(balance_score)
        
        return np.mean(balances)
    
    def _calculate_consistency(self, subgroup_samples: List[List[str]]) -> float:
        """
        计算一致性
        
        检查子群体内部是否存在不合理的模式
        """
        if len(subgroup_samples) < 2:
            return 1.0
        
        # 计算样本间的相似度
        similarities = []
        for i in range(min(len(subgroup_samples), 10)):
            for j in range(i + 1, min(len(subgroup_samples), 10)):
                similarity = sum(subgroup_samples[i][k] == subgroup_samples[j][k] 
                               for k in range(len(self.attr_names)))
                similarities.append(similarity / len(self.attr_names))
        
        if len(similarities) == 0:
            return 1.0
        
        # 一致性：相似度的方差不应过大
        consistency = 1.0 - np.std(similarities)
        return np.clip(consistency, 0.0, 1.0)
    
    def _identify_issues(self, diversity: float, concentration: float, 
                        balance: float) -> List[str]:
        """识别质量问题"""
        issues = []
        
        if diversity < self.quality_thresholds['diversity']:
            issues.append('低多样性：子群体内部缺乏变化')
        
        if concentration > self.quality_thresholds['concentration']:
            issues.append('高集中度：某些值过度集中')
        
        if balance < self.quality_thresholds['balance']:
            issues.append('不平衡：分布严重偏离期望')
        
        return issues
    
    def batch_evaluate(self, samples: List[List[str]], 
                      subgroup_conditions: List[Dict[str, str]]) -> Dict:
        """批量评估多个子群体"""
        results = {}
        for i, condition in enumerate(subgroup_conditions):
            condition_name = '&'.join([f"{k}={v}" for k, v in condition.items()])
            results[condition_name] = self.evaluate_subgroup(samples, condition)
        
        # 汇总报告
        avg_quality = np.mean([r['quality_score'] for r in results.values()])
        pass_rate = np.mean([r['passes_quality'] for r in results.values()])
        
        return {
            'subgroups': results,
            'average_quality': avg_quality,
            'pass_rate': pass_rate,
            'overall_passes': pass_rate >= 0.8
        }


# =========================
# 4. 迭代采样系统
# =========================

class IterativeSamplingSystem:
    """
    迭代采样系统
    
    整合配额退火和子群体判别，循环优化采样分布
    """
    
    def __init__(self, attr_names: List[str] = None, levels_list: List[str] = None):
        # 初始化真实分布数据
        self._initialize_distributions()
        
        self.attr_names = attr_names or [ATTR_A, ATTR_B, ATTR_C]
        self.levels = levels_list or levels
        
        self.quota_manager = QuotaManager()
        self.annealing = QuotaAnnealing()
        self.discriminator = SubgroupDiscriminator(self.attr_names, self.levels)
        
        self.iteration_history = []
        self.best_samples = None
        self.best_score = 0.0
    
    def _initialize_distributions(self):
        """初始化全局分布变量"""
        global P_A_2024, P_B_2024, P_C_2024, P_BC_2024, P_AB_2024, P_AC_2024
        
        # 调用初始化函数
        initialize_distributions_from_real_data()
        
        # 从模块导入已初始化的变量
        import make_finetune_datasets as mfd
        P_A_2024 = mfd.P_A_2024
        P_B_2024 = mfd.P_B_2024
        P_C_2024 = mfd.P_C_2024
        P_BC_2024 = mfd.P_BC_2024
        P_AB_2024 = mfd.P_AB_2024
        P_AC_2024 = mfd.P_AC_2024
    
    def set_quotas(self, quota_configs: List[Dict]):
        """
        设置配额
        
        Args:
            quota_configs: 配额配置列表
                例如: [
                    {'type': 'single', 'attr': 'age', 'value': '50', 'target': 0.2},
                    {'type': 'multi', 'dimensions': {'age': '50', 'gender': 'male'}, 'target': 0.1}
                ]
        """
        for config in quota_configs:
            if config['type'] == 'single':
                self.quota_manager.add_single_dim_quota(
                    attr=config['attr'],
                    value=config['value'],
                    target_prop=config['target'],
                    tolerance=config.get('tolerance', 0.05),
                    weight=config.get('weight', 1.0),
                    priority=config.get('priority', 1)
                )
            elif config['type'] == 'multi':
                self.quota_manager.add_multi_dim_quota(
                    dimensions=config['dimensions'],
                    target_prop=config['target'],
                    tolerance=config.get('tolerance', 0.05),
                    weight=config.get('weight', 1.0),
                    priority=config.get('priority', 1)
                )
    
    def _calculate_actual_proportions(self, samples: List[List[str]]) -> Dict[str, float]:
        """计算样本的实际配额比例"""
        proportions = {}
        total = len(samples)
        
        for quota in self.quota_manager.quotas:
            count = 0
            attr_indices = {attr: i for i, attr in enumerate(self.attr_names)}
            
            for sample in samples:
                match = True
                for attr, value in quota.dimensions.items():
                    if attr in attr_indices:
                        idx = attr_indices[attr]
                        if sample[idx] != value:
                            match = False
                            break
                if match:
                    count += 1
            
            proportions[quota.name] = count / total if total > 0 else 0.0
        
        return proportions
    
    def _adjust_sampling_distribution(self, Pa: pd.Series, Pb: pd.Series, Pc: pd.Series,
                                     iteration: int) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        根据配额权重调整采样分布
        
        使用加权混合：基准分布 + 配额导向分布
        """
        # 复制基准分布
        Pa_adj = Pa.copy()
        Pb_adj = Pb.copy()
        Pc_adj = Pc.copy()
        
        # 根据配额调整
        for quota in self.quota_manager.quotas:
            if len(quota.dimensions) == 1:  # 仅处理单维度配额
                attr, value = list(quota.dimensions.items())[0]
                weight = quota.weight * quota.priority / 5  # 考虑优先级
                
                # 确保value在分布中存在
                if value not in Pa_adj.index:
                    continue
                
                # 找到对应的分布
                if attr == self.attr_names[0]:
                    Pa_adj[value] += weight * (quota.target_proportion - Pa_adj[value]) * 0.2
                elif attr == self.attr_names[1]:
                    Pb_adj[value] += weight * (quota.target_proportion - Pb_adj[value]) * 0.2
                elif attr == self.attr_names[2]:
                    Pc_adj[value] += weight * (quota.target_proportion - Pc_adj[value]) * 0.2
        
        # 归一化
        Pa_adj = normalize_series(Pa_adj)
        Pb_adj = normalize_series(Pb_adj)
        Pc_adj = normalize_series(Pc_adj)
        
        return Pa_adj, Pb_adj, Pc_adj
    
    def run_iteration(self, n_samples: int = 100, max_iterations: int = 20,
                     convergence_threshold: float = 0.02,
                     enable_subgroup_check: bool = True) -> Dict:
        """
        运行迭代采样
        
        Args:
            n_samples: 每次迭代生成的样本数
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值
            enable_subgroup_check: 是否启用子群体检查
            
        Returns:
            最终结果和报告
        """
        print(f"开始迭代采样系统...")
        print(f"配额数量: {len(self.quota_manager.quotas)}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"收敛阈值: {convergence_threshold}")
        
        # 从模块获取分布数据
        import make_finetune_datasets as mfd
        
        for iteration in range(max_iterations):
            print(f"\n=== 迭代 {iteration + 1}/{max_iterations} ===")
            
            # 1. 调整采样分布
            Pa_adj, Pb_adj, Pc_adj = self._adjust_sampling_distribution(
                mfd.P_A_2024, mfd.P_B_2024, mfd.P_C_2024, iteration
            )
            
            # 2. 使用IPF构建联合分布
            Q_adj = ipf_with_tables(
                Pa_adj, Pb_adj, Pc_adj,
                Pbc=mfd.P_BC_2024, Pab=mfd.P_AB_2024, Pac=mfd.P_AC_2024,
                iters=800
            )
            
            # 3. 采样
            samples = sample_from_joint(Q_adj, n=n_samples, seed=None)
            
            # 4. 计算实际比例
            actual_proportions = self._calculate_actual_proportions(samples)
            
            # 5. 评估配额满足情况
            quota_satisfaction = {}
            total_deviation = 0.0
            for quota in self.quota_manager.quotas:
                actual = actual_proportions[quota.name]
                satisfied = quota.is_satisfied(actual)
                deviation = quota.deviation(actual)
                quota_satisfaction[quota.name] = {
                    'target': quota.target_proportion,
                    'actual': actual,
                    'satisfied': satisfied,
                    'deviation': deviation,
                    'weight': quota.weight
                }
                total_deviation += deviation * quota.weight
            
            # 6. 子群体质量评估（如果启用）
            subgroup_report = None
            if enable_subgroup_check:
                # 定义要检查的子群体
                subgroup_conditions = [
                    {quota.dimensions for quota in self.quota_manager.quotas if len(quota.dimensions) == 1}
                ]
                subgroup_conditions = [dict(quota.dimensions) for quota in self.quota_manager.quotas][:5]
                if subgroup_conditions:
                    subgroup_report = self.discriminator.batch_evaluate(samples, subgroup_conditions)
            
            # 7. 配额退火
            self.annealing.anneal_quota_weights(
                self.quota_manager, actual_proportions, iteration
            )
            
            # 8. 记录历史
            iteration_info = {
                'iteration': iteration + 1,
                'total_deviation': total_deviation,
                'quota_satisfaction': quota_satisfaction,
                'subgroup_report': subgroup_report,
                'temperature': self.annealing.temperature,
                'samples_preview': samples[:3]  # 保存前3个样本作为预览
            }
            self.iteration_history.append(iteration_info)
            
            # 9. 更新最佳结果
            quality_score = self._calculate_overall_score(
                quota_satisfaction, subgroup_report
            )
            if quality_score > self.best_score:
                self.best_score = quality_score
                self.best_samples = samples
                print(f"✨ 发现更优解! 质量得分: {quality_score:.4f}")
            
            # 10. 打印当前状态
            self._print_iteration_status(iteration_info, quality_score)
            
            # 11. 检查收敛
            if total_deviation < convergence_threshold:
                print(f"\n✅ 收敛! 总偏差 {total_deviation:.4f} < {convergence_threshold}")
                break
        
        # 生成最终报告
        final_report = self._generate_final_report()
        return final_report
    
    def _calculate_overall_score(self, quota_satisfaction: Dict, 
                                subgroup_report: Optional[Dict]) -> float:
        """计算整体质量得分"""
        # 配额满足得分
        quota_scores = []
        for info in quota_satisfaction.values():
            if info['deviation'] < 0.05:
                quota_scores.append(1.0)
            else:
                quota_scores.append(max(0, 1.0 - info['deviation'] / 0.2))
        quota_score = np.mean(quota_scores) if quota_scores else 0.0
        
        # 子群体质量得分
        subgroup_score = 0.8  # 默认值
        if subgroup_report:
            subgroup_score = subgroup_report.get('average_quality', 0.8)
        
        # 综合得分
        overall_score = 0.7 * quota_score + 0.3 * subgroup_score
        return overall_score
    
    def _print_iteration_status(self, iteration_info: Dict, quality_score: float):
        """打印迭代状态"""
        print(f"质量得分: {quality_score:.4f}")
        print(f"温度: {iteration_info['temperature']:.3f}")
        print(f"总偏差: {iteration_info['total_deviation']:.4f}")
        
        print("\n配额满足情况:")
        for quota_name, info in iteration_info['quota_satisfaction'].items():
            status = "✓" if info['satisfied'] else "✗"
            print(f"  {status} {quota_name}: {info['actual']:.3f} (目标: {info['target']:.3f}, "
                  f"偏差: {info['deviation']:.3f}, 权重: {info['weight']:.2f})")
        
        if iteration_info['subgroup_report']:
            sr = iteration_info['subgroup_report']
            print(f"\n子群体质量: 平均 {sr['average_quality']:.3f}, 通过率 {sr['pass_rate']:.2%}")
    
    def _generate_final_report(self) -> Dict:
        """生成最终报告"""
        report = {
            'total_iterations': len(self.iteration_history),
            'best_score': self.best_score,
            'best_samples': self.best_samples,
            'best_samples_count': len(self.best_samples) if self.best_samples else 0,
            'convergence_achieved': self.iteration_history[-1]['total_deviation'] < 0.02 if self.iteration_history else False,
            'final_quota_satisfaction': self.iteration_history[-1]['quota_satisfaction'] if self.iteration_history else {},
            'annealing_report': self.annealing.get_annealing_report(),
            'iteration_history': self.iteration_history
        }
        return report
    
    def save_results(self, output_path: str):
        """保存结果到文件"""
        report = self._generate_final_report()
        
        # 转换不可序列化的对象
        save_report = {
            'total_iterations': report['total_iterations'],
            'best_score': report['best_score'],
            'best_samples_count': report['best_samples_count'],
            'convergence_achieved': report['convergence_achieved'],
            'final_quota_satisfaction': report['final_quota_satisfaction'],
            'annealing_report': report['annealing_report']
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(save_report, f, ensure_ascii=False, indent=2)
        
        print(f"\n结果已保存到: {output_path}")
    
    def export_best_samples(self, output_path: str):
        """导出最佳样本"""
        if self.best_samples is None:
            print("没有可导出的最佳样本")
            return
        
        # 转换为DataFrame
        df = pd.DataFrame(self.best_samples, columns=self.attr_names)
        df.to_csv(output_path, index=False)
        print(f"最佳样本已导出到: {output_path}")


# =========================
# 5. 使用示例
# =========================

def example_basic_usage():
    """基础使用示例"""
    print("=" * 60)
    print("配额退火系统 - 基础使用示例")
    print("=" * 60)
    
    # 创建系统实例
    system = IterativeSamplingSystem()
    
    # 设置配额（示例：希望某些属性组合达到特定比例）
    quota_configs = [
        # 单维度配额
        {'type': 'single', 'attr': ATTR_A, 'value': '3', 'target': 0.25, 'tolerance': 0.05, 'priority': 3},
        {'type': 'single', 'attr': ATTR_B, 'value': '3', 'target': 0.20, 'tolerance': 0.05, 'priority': 2},
        {'type': 'single', 'attr': ATTR_C, 'value': '2', 'target': 0.35, 'tolerance': 0.05, 'priority': 1},
        
        # 多维度配额（如果需要）
        # {'type': 'multi', 'dimensions': {ATTR_A: '3', ATTR_B: '3'}, 'target': 0.10, 'priority': 4}
    ]
    
    system.set_quotas(quota_configs)
    
    # 运行迭代优化
    result = system.run_iteration(
        n_samples=200,
        max_iterations=15,
        convergence_threshold=0.02,
        enable_subgroup_check=True
    )
    
    # 打印最终报告
    print("\n" + "=" * 60)
    print("最终报告")
    print("=" * 60)
    print(f"总迭代次数: {result['total_iterations']}")
    print(f"最佳质量得分: {result['best_score']:.4f}")
    print(f"是否收敛: {result['convergence_achieved']}")
    print(f"最佳样本数量: {result['best_samples_count']}")
    
    # 保存结果
    system.save_results('/home/cyyuan/FAS_PRO/sft/teacher_distribution/quota_annealing_report.json')
    system.export_best_samples('/home/cyyuan/FAS_PRO/sft/teacher_distribution/best_samples.csv')
    
    return system, result


def example_advanced_usage():
    """高级使用示例：自定义配额和子群体检查"""
    print("=" * 60)
    print("配额退火系统 - 高级使用示例")
    print("=" * 60)
    
    system = IterativeSamplingSystem()
    
    # 更复杂的配额设置
    quota_configs = [
        # 不同优先级的配额
        {'type': 'single', 'attr': ATTR_A, 'value': '1', 'target': 0.50, 'tolerance': 0.03, 
         'weight': 1.5, 'priority': 5},  # 高优先级
        {'type': 'single', 'attr': ATTR_B, 'value': '2', 'target': 0.30, 'tolerance': 0.05, 
         'weight': 1.0, 'priority': 3},  # 中优先级
        {'type': 'single', 'attr': ATTR_C, 'value': '3', 'target': 0.15, 'tolerance': 0.05, 
         'weight': 0.8, 'priority': 1},  # 低优先级
        
        # 多维度配额
        {'type': 'multi', 'dimensions': {ATTR_A: '1', ATTR_B: '2'}, 
         'target': 0.12, 'tolerance': 0.03, 'priority': 4}
    ]
    
    system.set_quotas(quota_configs)
    
    # 运行迭代（更多迭代次数，更严格的收敛条件）
    result = system.run_iteration(
        n_samples=500,
        max_iterations=30,
        convergence_threshold=0.01,
        enable_subgroup_check=True
    )
    
    print("\n高级分析:")
    print(f"退火温度历史: {[h['temperature'] for h in result['iteration_history'][-5:]]}")
    print(f"偏差历史: {[h['total_deviation'] for h in result['iteration_history'][-5:]]}")
    
    return system, result


if __name__ == "__main__":
    # 运行基础示例
    system, result = example_basic_usage()
    
    print("\n" + "=" * 60)
    print("运行完成!")
    print("=" * 60)
    
    # 如需运行高级示例，取消下面的注释
    # system_adv, result_adv = example_advanced_usage()

