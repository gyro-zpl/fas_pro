"""
整合Dify API调用和配额退火系统

使用场景：
1. 通过Dify API调用大模型生成样本
2. 使用配额退火系统检查和优化生成质量
3. 迭代修正直到满足配额要求
"""

import os
import sys
import json
import csv
import requests
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

# 导入原有的Dify相关函数
from DIFY_none_RAG import (
    extract_rows, 
    validate_rows,
    dify_request_non_stream
)

# 重写clean_model_output以更好地处理各种格式
def clean_model_output(text: str):
    """增强版：清理模型输出并提取数据"""
    import re
    
    s = (text or "").strip()
    
    # 打印原始输出以便调试
    print(f"\n[DEBUG] 原始模型输出（前200字符）:\n{s[:200]}")
    
    # 移除markdown代码块
    if s.startswith("```json"):
        s = s[7:]
    elif s.startswith("```"):
        s = s[3:]
    if s.endswith("```"):
        s = s[:-3]
    s = s.strip()
    
    # 1) 尝试直接解析JSON
    try:
        data = json.loads(s)
        print(f"[DEBUG] 直接解析成功，类型: {type(data)}")
        return data
    except Exception as e:
        print(f"[DEBUG] 直接解析失败: {e}")
    
    # 2) 查找JSON数组 [[...], [...]]
    array_pattern = r'\[\s*\[.*?\]\s*\]'
    matches = re.findall(array_pattern, s, re.DOTALL)
    if matches:
        for m in matches:
            try:
                data = json.loads(m)
                print(f"[DEBUG] 找到JSON数组，长度: {len(data)}")
                return data
            except Exception:
                continue
    
    # 3) 提取所有['x','x','x']格式的行
    # 匹配 ['1', '2', '3'] 或 ["1", "2", "3"] 或 ['1','2','3']
    row_pattern = r"\[\s*['\"](\d+)['\"]\s*,\s*['\"](\d+)['\"]\s*,\s*['\"](\d+)['\"]\s*\]"
    rows = re.findall(row_pattern, s)
    if rows:
        result = [[r[0], r[1], r[2]] for r in rows]
        print(f"[DEBUG] 正则提取到 {len(result)} 行")
        return result
    
    # 4) 尝试更宽松的模式：可能没有引号
    row_pattern2 = r"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"
    rows = re.findall(row_pattern2, s)
    if rows:
        result = [[r[0], r[1], r[2]] for r in rows]
        print(f"[DEBUG] 宽松模式提取到 {len(result)} 行")
        return result
    
    # 5) 查找任何看起来像数组的结构
    lines = s.split('\n')
    result = []
    for line in lines:
        line = line.strip()
        if '[' in line and ']' in line:
            # 尝试提取这一行
            try:
                # 提取方括号内的内容
                match = re.search(r'\[(.*?)\]', line)
                if match:
                    content = match.group(1)
                    # 分割并清理
                    parts = [p.strip().strip("'\"") for p in content.split(',')]
                    if len(parts) == 3 and all(p in ['1', '2', '3'] for p in parts):
                        result.append(parts)
            except Exception:
                continue
    
    if result:
        print(f"[DEBUG] 逐行解析得到 {len(result)} 行")
        return result
    
    # 如果都失败了，打印更多调试信息
    print(f"[DEBUG] 所有解析方法都失败了")
    print(f"[DEBUG] 完整输出:\n{s}")
    raise ValueError("Failed to extract valid JSON data from model output")

# 导入配额退火系统
sys.path.append('/home/cyyuan/FAS_PRO/sft/teacher_distribution')
from quota_annealing_system import IterativeSamplingSystem


class DifyQuotaGenerator:
    """
    整合Dify API和配额退火的生成器
    
    工作流程：
    1. 设置目标配额
    2. 调用Dify生成初始批次
    3. 评估当前样本是否满足配额
    4. 如果不满足，根据偏差调整prompt重新生成
    5. 迭代直到收敛或达到最大次数
    """
    
    def __init__(self, columns: List[str], mission: str):
        """
        初始化生成器
        
        Args:
            columns: 列名列表，如 ['natcity', 'natenvir', 'natheal']
            mission: 任务描述
        """
        self.columns = columns
        self.mission = mission
        self.conversation_id = None
        
        # 初始化配额系统（仅用于评估和指导）
        self.quota_system = None
        self.quota_configs = []
        
        # 记录历史
        self.history = []
        self.all_samples = []
    
    def set_quotas(self, quota_configs: List[Dict]):
        """
        设置配额要求
        
        Args:
            quota_configs: 配额配置列表
                例如: [
                    {'type': 'single', 'attr': 'natcity', 'value': '3', 
                     'target': 0.20, 'tolerance': 0.05, 'priority': 4}
                ]
        """
        self.quota_configs = quota_configs
        print(f"\n✅ 已设置 {len(quota_configs)} 个配额:")
        for qc in quota_configs:
            if qc['type'] == 'single':
                print(f"   - {qc['attr']}='{qc['value']}' → 目标{qc['target']:.1%} (±{qc['tolerance']:.1%})")
            else:
                dims = '&'.join([f"{k}='{v}'" for k, v in qc['dimensions'].items()])
                print(f"   - {dims} → 目标{qc['target']:.1%} (±{qc['tolerance']:.1%})")
    
    def _calculate_current_proportions(self, samples: List[List[str]]) -> Dict[str, float]:
        """计算当前样本的配额比例"""
        if not samples:
            return {}
        
        proportions = {}
        total = len(samples)
        
        for qc in self.quota_configs:
            if qc['type'] == 'single':
                # 单维度配额
                attr_idx = self.columns.index(qc['attr'])
                count = sum(1 for s in samples if s[attr_idx] == qc['value'])
                quota_name = f"{qc['attr']}={qc['value']}"
                proportions[quota_name] = count / total
                
            elif qc['type'] == 'multi':
                # 多维度配额
                count = 0
                for sample in samples:
                    match = True
                    for attr, value in qc['dimensions'].items():
                        attr_idx = self.columns.index(attr)
                        if sample[attr_idx] != value:
                            match = False
                            break
                    if match:
                        count += 1
                quota_name = '&'.join([f"{k}={v}" for k, v in qc['dimensions'].items()])
                proportions[quota_name] = count / total
        
        return proportions
    
    def _evaluate_satisfaction(self, proportions: Dict[str, float]) -> Tuple[bool, Dict]:
        """评估配额满足情况"""
        satisfaction = {}
        all_satisfied = True
        
        for qc in self.quota_configs:
            if qc['type'] == 'single':
                quota_name = f"{qc['attr']}={qc['value']}"
            else:
                quota_name = '&'.join([f"{k}={v}" for k, v in qc['dimensions'].items()])
            
            actual = proportions.get(quota_name, 0.0)
            target = qc['target']
            tolerance = qc['tolerance']
            
            satisfied = abs(actual - target) <= tolerance
            deviation = abs(actual - target)
            
            satisfaction[quota_name] = {
                'target': target,
                'actual': actual,
                'deviation': deviation,
                'satisfied': satisfied,
                'tolerance': tolerance,
                'priority': qc.get('priority', 3)
            }
            
            if not satisfied:
                all_satisfied = False
        
        return all_satisfied, satisfaction
    
    def _generate_adaptive_prompt(self, base_batch_size: int, iteration: int, 
                                  satisfaction: Dict = None) -> str:
        """
        根据当前配额满足情况生成自适应的prompt
        
        如果某些配额未满足，在prompt中强调需要增加的类别
        """
        prompt_parts = [
            f"Mission: {self.mission}.",
            f"Generate batch No.{iteration}.",
            f"Sample batch size is {base_batch_size}."
        ]
        
        # 如果有配额信息，添加指导
        if satisfaction:
            # 找出未满足的高优先级配额
            unsatisfied = [
                (name, info) for name, info in satisfaction.items() 
                if not info['satisfied'] and info['priority'] >= 3
            ]
            
            if unsatisfied:
                # 按偏差大小排序
                unsatisfied.sort(key=lambda x: x[1]['deviation'], reverse=True)
                
                # 添加指导信息
                guidance = []
                for name, info in unsatisfied[:2]:  # 只强调最重要的2个
                    if info['actual'] < info['target']:
                        # 需要增加这个类别
                        diff_pct = (info['target'] - info['actual']) * 100
                        guidance.append(
                            f"increase samples with {name} by approximately {diff_pct:.1f}%"
                        )
                
                if guidance:
                    prompt_parts.append(
                        f"Adjust distribution: {'; '.join(guidance)}."
                    )
        
        return " ".join(prompt_parts)
    
    def generate_with_quota_annealing(
        self, 
        target_total: int = 400,
        batch_size: int = 50,
        max_iterations: int = 20,
        convergence_threshold: float = 0.03
    ) -> Tuple[List[List[str]], Dict]:
        """
        使用配额退火策略生成样本
        
        Args:
            target_total: 目标总样本数
            batch_size: 每批生成的样本数
            max_iterations: 最大迭代次数
            convergence_threshold: 收敛阈值（总偏差<此值即收敛）
        
        Returns:
            (samples, report): 生成的样本列表和详细报告
        """
        print("\n" + "=" * 70)
        print("🚀 开始配额退火生成")
        print("=" * 70)
        print(f"目标样本数: {target_total}")
        print(f"每批大小: {batch_size}")
        print(f"最大迭代: {max_iterations}")
        print(f"收敛阈值: {convergence_threshold}")
        
        all_samples = []
        iteration = 0
        converged = False
        
        for iteration in range(max_iterations):
            print(f"\n{'='*70}")
            print(f"📍 迭代 {iteration + 1}/{max_iterations}")
            print(f"{'='*70}")
            
            # 计算当前状态
            current_proportions = self._calculate_current_proportions(all_samples)
            all_satisfied, satisfaction = self._evaluate_satisfaction(current_proportions)
            
            # 计算总偏差
            total_deviation = sum(info['deviation'] for info in satisfaction.values())
            
            # 打印当前状态
            print(f"当前样本数: {len(all_samples)}")
            print(f"总偏差: {total_deviation:.4f}")
            
            if satisfaction:
                print("\n配额满足情况:")
                for quota_name, info in satisfaction.items():
                    status = "✓" if info['satisfied'] else "✗"
                    print(f"  {status} {quota_name}: {info['actual']:.3f} "
                          f"(目标: {info['target']:.3f}, 偏差: {info['deviation']:.3f})")
            
            # 检查是否收敛
            if len(all_samples) >= target_total and total_deviation < convergence_threshold:
                print(f"\n✅ 收敛! 总偏差 {total_deviation:.4f} < {convergence_threshold}")
                converged = True
                break
            
            # 如果样本数已够但未收敛，采用替换策略
            if len(all_samples) >= target_total:
                print(f"\n⚠️  样本数已达 {target_total}，但未收敛。采用替换策略...")
                # 移除一些样本，生成新的
                remove_count = min(batch_size, len(all_samples) // 4)
                all_samples = all_samples[:-remove_count]
                print(f"   移除 {remove_count} 个旧样本")
            
            # 计算本次需要生成的数量
            remaining = target_total - len(all_samples)
            current_batch_size = min(batch_size, remaining)
            
            if current_batch_size <= 0:
                current_batch_size = batch_size  # 替换模式
            
            # 生成自适应prompt
            prompt = self._generate_adaptive_prompt(
                current_batch_size, 
                iteration,
                satisfaction if iteration > 0 else None
            )
            
            print(f"\n📝 Prompt: {prompt[:150]}...")
            
            # 调用Dify API
            try:
                raw_response, self.conversation_id = dify_request_non_stream(
                    prompt, 
                    self.conversation_id
                )
                
                # 解析和验证
                data_obj = clean_model_output(raw_response) if isinstance(raw_response, str) else raw_response
                rows = extract_rows(data_obj)
                rows = validate_rows(rows)
                
                # 添加到总样本
                all_samples.extend(rows)
                
                print(f"✅ 生成 {len(rows)} 个样本 (累计: {len(all_samples)})")
                
                # 记录历史
                self.history.append({
                    'iteration': iteration + 1,
                    'prompt': prompt,
                    'generated': len(rows),
                    'total_samples': len(all_samples),
                    'total_deviation': total_deviation,
                    'satisfaction': satisfaction
                })
                
            except Exception as e:
                print(f"❌ 错误: {e}")
                continue
        
        # 如果达到最大迭代仍未收敛
        if not converged:
            print(f"\n⚠️  达到最大迭代次数 {max_iterations}，未完全收敛")
        
        # 生成最终报告
        final_proportions = self._calculate_current_proportions(all_samples)
        all_satisfied, final_satisfaction = self._evaluate_satisfaction(final_proportions)
        final_deviation = sum(info['deviation'] for info in final_satisfaction.values())
        
        report = {
            'total_iterations': iteration + 1,
            'converged': converged,
            'total_samples': len(all_samples),
            'final_deviation': final_deviation,
            'all_quotas_satisfied': all_satisfied,
            'final_satisfaction': final_satisfaction,
            'history': self.history
        }
        
        # 打印最终总结
        print("\n" + "=" * 70)
        print("📊 最终结果")
        print("=" * 70)
        print(f"总样本数: {len(all_samples)}")
        print(f"迭代次数: {iteration + 1}")
        print(f"是否收敛: {'是' if converged else '否'}")
        print(f"最终总偏差: {final_deviation:.4f}")
        print(f"配额全部满足: {'是' if all_satisfied else '否'}")
        
        print("\n最终配额满足情况:")
        for quota_name, info in final_satisfaction.items():
            status = "✅" if info['satisfied'] else "❌"
            print(f"  {status} {quota_name}: {info['actual']:.3f} "
                  f"(目标: {info['target']:.3f}, 偏差: {info['deviation']:.3f})")
        
        self.all_samples = all_samples
        return all_samples, report
    
    def save_to_csv(self, samples: List[List[str]], output_path: str):
        """保存样本到CSV"""
        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(self.columns)
            for row in samples:
                clean_row = [int(x) if str(x).isdigit() else x for x in row]
                writer.writerow(clean_row)
        print(f"\n💾 已保存 {len(samples)} 个样本到: {output_path}")
    
    def save_report(self, report: Dict, output_path: str):
        """保存详细报告"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"📄 已保存报告到: {output_path}")


def example_basic_usage():
    """示例1: 基础使用"""
    print("\n" + "🌟" * 35)
    print("示例1: 基础配额退火生成")
    print("🌟" * 35)
    
    # 创建生成器
    columns = ['natcity', 'natenvir', 'natheal']
    mission = "Generate samples representing public opinion on national spending"
    
    generator = DifyQuotaGenerator(columns=columns, mission=mission)
    
    # 设置配额
    quota_configs = [
        {'type': 'single', 'attr': 'natcity', 'value': '3', 
         'target': 0.20, 'tolerance': 0.05, 'priority': 4}
    ]
    generator.set_quotas(quota_configs)
    
    # 生成
    samples, report = generator.generate_with_quota_annealing(
        target_total=200,
        batch_size=50,
        max_iterations=10,
        convergence_threshold=0.03
    )
    
    # 保存
    generator.save_to_csv(samples, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_basic.csv')
    generator.save_report(report, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_basic_report.json')


def example_multi_quota():
    """示例2: 多配额使用"""
    print("\n" + "🌟" * 35)
    print("示例2: 多配额退火生成")
    print("🌟" * 35)
    
    columns = ['natcity', 'natenvir', 'natheal']
    mission = "Generate samples with specific distribution targets"
    
    generator = DifyQuotaGenerator(columns=columns, mission=mission)
    
    # 设置多个配额
    quota_configs = [
        # 单维度配额
        {'type': 'single', 'attr': 'natcity', 'value': '3', 
         'target': 0.18, 'tolerance': 0.04, 'priority': 5},
        {'type': 'single', 'attr': 'natenvir', 'value': '2', 
         'target': 0.30, 'tolerance': 0.05, 'priority': 3},
        
        # 多维度配额
        {'type': 'multi', 'dimensions': {'natcity': '3', 'natenvir': '2'}, 
         'target': 0.08, 'tolerance': 0.03, 'priority': 4}
    ]
    generator.set_quotas(quota_configs)
    
    # 生成
    samples, report = generator.generate_with_quota_annealing(
        target_total=400,
        batch_size=60,
        max_iterations=15,
        convergence_threshold=0.02
    )
    
    # 保存
    generator.save_to_csv(samples, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_multi.csv')
    generator.save_report(report, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_multi_report.json')


def example_with_custom_mission():
    """示例3: 自定义任务"""
    print("\n" + "🌟" * 35)
    print("示例3: 自定义任务配额生成")
    print("🌟" * 35)
    
    columns = ['natcity', 'natenvir', 'natheal']
    
    # 自定义任务描述
    mission = """Generate synthetic survey responses about national spending priorities. 
    Each response should reflect realistic opinion patterns from American public opinion surveys."""
    
    generator = DifyQuotaGenerator(columns=columns, mission=mission)
    
    # 根据真实数据分布设置配额
    quota_configs = [
        {'type': 'single', 'attr': 'natcity', 'value': '1', 
         'target': 0.55, 'tolerance': 0.06, 'priority': 4},
        {'type': 'single', 'attr': 'natenvir', 'value': '1', 
         'target': 0.70, 'tolerance': 0.06, 'priority': 4},
        {'type': 'single', 'attr': 'natheal', 'value': '1', 
         'target': 0.68, 'tolerance': 0.06, 'priority': 4}
    ]
    generator.set_quotas(quota_configs)
    
    # 生成
    samples, report = generator.generate_with_quota_annealing(
        target_total=500,
        batch_size=80,
        max_iterations=20,
        convergence_threshold=0.025
    )
    
    # 保存
    generator.save_to_csv(samples, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_custom.csv')
    generator.save_report(report, '/home/cyyuan/FAS_PRO/data/gss/test_results/dify_quota_custom_report.json')


def quick_test():
    """快速测试"""
    print("\n" + "⚡" * 35)
    print("快速测试")
    print("⚡" * 35)
    
    columns = ['natcity', 'natenvir', 'natheal']
    mission = "Quick test"
    
    generator = DifyQuotaGenerator(columns=columns, mission=mission)
    
    quota_configs = [
        {'type': 'single', 'attr': 'natcity', 'value': '3', 
         'target': 0.20, 'tolerance': 0.08, 'priority': 3}
    ]
    generator.set_quotas(quota_configs)
    
    samples, report = generator.generate_with_quota_annealing(
        target_total=50,
        batch_size=20,
        max_iterations=5,
        convergence_threshold=0.05
    )
    
    print(f"\n✅ 快速测试完成! 生成 {len(samples)} 个样本")
    return samples, report


def run_from_config(mission, config_file, target_total=400, batch_size=50, 
                    max_iterations=15, output_csv=None):
    """从配置文件运行"""
    # 从环境变量读取列名
    columns_str = os.environ.get("COLUMNS", '["natcity", "natenvir", "natheal"]')
    columns = json.loads(columns_str)
    
    # 创建生成器
    generator = DifyQuotaGenerator(columns=columns, mission=mission)
    
    # 加载配额配置
    with open(config_file, 'r', encoding='utf-8') as f:
        quota_configs = json.load(f)
    
    generator.set_quotas(quota_configs)
    
    # 生成
    samples, report = generator.generate_with_quota_annealing(
        target_total=target_total,
        batch_size=batch_size,
        max_iterations=max_iterations
    )
    
    # 保存
    if output_csv:
        generator.save_to_csv(samples, output_csv)
        report_path = output_csv.replace('.csv', '_report.json')
        generator.save_report(report, report_path)
    
    return samples, report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Dify + 配额退火系统')
    parser.add_argument('mission', type=str, nargs='?', default=None,
                       help='任务名称（如ehc、edc）')
    parser.add_argument('--mode', type=str, default=None,
                       choices=['quick', 'basic', 'multi', 'custom'],
                       help='运行模式（如果不指定mission）')
    parser.add_argument('--config', type=str,
                       help='配额配置文件路径')
    parser.add_argument('--target_total', type=int, default=400,
                       help='目标样本数')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='每批大小')
    parser.add_argument('--max_iterations', type=int, default=15,
                       help='最大迭代次数')
    parser.add_argument('--output_csv', type=str,
                       help='输出CSV路径')
    
    args = parser.parse_args()
    
    # 如果指定了mission和config，使用配置文件模式
    if args.mission and args.config:
        run_from_config(
            mission=args.mission,
            config_file=args.config,
            target_total=args.target_total,
            batch_size=args.batch_size,
            max_iterations=args.max_iterations,
            output_csv=args.output_csv
        )
    # 否则使用示例模式
    elif args.mode:
        if args.mode == 'quick':
            quick_test()
        elif args.mode == 'basic':
            example_basic_usage()
        elif args.mode == 'multi':
            example_multi_quota()
        elif args.mode == 'custom':
            example_with_custom_mission()
    else:
        # 默认快速测试
        quick_test()
    
    print("\n" + "🎉" * 35)
    print("运行完成!")
    print("🎉" * 35)

