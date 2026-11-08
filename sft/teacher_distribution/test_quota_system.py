"""
配额退火系统测试脚本

快速验证系统各个组件是否正常工作
"""

import sys
import numpy as np
import pandas as pd

def test_imports():
    """测试1: 验证所有必要的导入"""
    print("\n" + "=" * 60)
    print("测试1: 验证导入")
    print("=" * 60)
    
    try:
        from quota_annealing_system import (
            IterativeSamplingSystem,
            QuotaManager,
            Quota,
            QuotaAnnealing,
            SubgroupDiscriminator
        )
        print("✅ 所有模块导入成功")
        return True
    except Exception as e:
        print(f"❌ 导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_quota_creation():
    """测试2: 验证配额创建"""
    print("\n" + "=" * 60)
    print("测试2: 配额创建")
    print("=" * 60)
    
    try:
        from quota_annealing_system import Quota, QuotaManager
        
        # 创建单个配额
        quota = Quota(
            name="test_quota",
            dimensions={'attr_a': '3'},
            target_proportion=0.25,
            tolerance=0.05,
            weight=1.0,
            priority=3
        )
        print(f"✅ 单个配额创建成功: {quota}")
        
        # 创建配额管理器
        manager = QuotaManager()
        manager.add_quota(quota)
        manager.add_single_dim_quota('attr_b', '2', 0.30, priority=2)
        manager.add_multi_dim_quota({'attr_a': '3', 'attr_b': '2'}, 0.10, priority=4)
        
        print(f"✅ 配额管理器创建成功，共 {len(manager.quotas)} 个配额")
        for q in manager.quotas:
            print(f"   - {q}")
        
        return True
    except Exception as e:
        print(f"❌ 配额创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_annealing():
    """测试3: 验证退火机制"""
    print("\n" + "=" * 60)
    print("测试3: 退火机制")
    print("=" * 60)
    
    try:
        from quota_annealing_system import QuotaAnnealing, QuotaManager, Quota
        
        # 创建退火器
        annealing = QuotaAnnealing(
            initial_temperature=1.0,
            cooling_rate=0.95,
            min_temperature=0.1
        )
        print(f"✅ 退火器创建成功，初始温度: {annealing.temperature}")
        
        # 创建测试配额
        manager = QuotaManager()
        manager.add_single_dim_quota('attr_a', '3', 0.25, priority=3)
        
        # 模拟退火过程
        for i in range(3):
            actual_props = {'attr_a=3': 0.20 + i * 0.02}
            weights = annealing.anneal_quota_weights(manager, actual_props, i)
            print(f"   迭代{i+1}: 温度={annealing.temperature:.3f}, 权重={weights}")
        
        print("✅ 退火机制运行正常")
        return True
    except Exception as e:
        print(f"❌ 退火机制测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_discriminator():
    """测试4: 验证子群体判别器"""
    print("\n" + "=" * 60)
    print("测试4: 子群体判别器")
    print("=" * 60)
    
    try:
        from quota_annealing_system import SubgroupDiscriminator
        
        # 创建判别器
        discriminator = SubgroupDiscriminator(
            attr_names=['attr_a', 'attr_b', 'attr_c'],
            levels=['1', '2', '3']
        )
        print("✅ 判别器创建成功")
        
        # 创建测试样本
        np.random.seed(42)
        samples = []
        for _ in range(50):
            sample = [
                np.random.choice(['1', '2', '3']),
                np.random.choice(['1', '2', '3']),
                np.random.choice(['1', '2', '3'])
            ]
            samples.append(sample)
        
        # 评估子群体
        result = discriminator.evaluate_subgroup(
            samples,
            {'attr_a': '3'}
        )
        
        print(f"✅ 子群体评估成功")
        print(f"   样本量: {result['size']}")
        print(f"   比例: {result['proportion']:.3f}")
        print(f"   质量得分: {result['quality_score']:.3f}")
        print(f"   多样性: {result['diversity']:.3f}")
        
        return True
    except Exception as e:
        print(f"❌ 判别器测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_basic_system():
    """测试5: 验证基础系统运行"""
    print("\n" + "=" * 60)
    print("测试5: 基础系统运行（简化版）")
    print("=" * 60)
    
    try:
        from quota_annealing_system import IterativeSamplingSystem
        from make_finetune_datasets import ATTR_A, ATTR_B, ATTR_C
        
        print("正在创建系统...")
        system = IterativeSamplingSystem()
        print("✅ 系统创建成功")
        
        # 设置简单配额
        print("正在设置配额...")
        quota_configs = [
            {'type': 'single', 'attr': ATTR_A, 'value': '3', 
             'target': 0.25, 'tolerance': 0.08, 'priority': 3}
        ]
        system.set_quotas(quota_configs)
        print(f"✅ 配额设置成功，共 {len(system.quota_manager.quotas)} 个配额")
        
        # 运行简短迭代
        print("正在运行系统（2次迭代，快速测试）...")
        result = system.run_iteration(
            n_samples=50,  # 小样本
            max_iterations=2,  # 少迭代
            convergence_threshold=0.10,  # 宽松阈值
            enable_subgroup_check=False  # 禁用子群体检查以加快速度
        )
        
        print("✅ 系统运行成功!")
        print(f"   迭代次数: {result['total_iterations']}")
        print(f"   最佳得分: {result['best_score']:.4f}")
        print(f"   样本数: {result['best_samples_count']}")
        
        return True
    except Exception as e:
        print(f"❌ 系统运行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_export():
    """测试6: 验证数据导出"""
    print("\n" + "=" * 60)
    print("测试6: 数据导出")
    print("=" * 60)
    
    try:
        from quota_annealing_system import IterativeSamplingSystem
        from make_finetune_datasets import ATTR_A
        import tempfile
        import os
        
        system = IterativeSamplingSystem()
        
        quota_configs = [
            {'type': 'single', 'attr': ATTR_A, 'value': '3', 
             'target': 0.25, 'tolerance': 0.10, 'priority': 3}
        ]
        system.set_quotas(quota_configs)
        
        result = system.run_iteration(
            n_samples=30,
            max_iterations=2,
            convergence_threshold=0.15,
            enable_subgroup_check=False
        )
        
        # 测试导出
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = os.path.join(tmpdir, 'test_report.json')
            samples_path = os.path.join(tmpdir, 'test_samples.csv')
            
            system.save_results(report_path)
            system.export_best_samples(samples_path)
            
            # 验证文件是否存在
            if os.path.exists(report_path) and os.path.exists(samples_path):
                print("✅ 数据导出成功")
                print(f"   报告文件: {os.path.getsize(report_path)} 字节")
                print(f"   样本文件: {os.path.getsize(samples_path)} 字节")
                return True
            else:
                print("❌ 导出文件未找到")
                return False
                
    except Exception as e:
        print(f"❌ 数据导出测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "🧪" * 30)
    print("配额退火系统 - 完整测试套件")
    print("🧪" * 30)
    
    tests = [
        ("导入测试", test_imports),
        ("配额创建", test_quota_creation),
        ("退火机制", test_annealing),
        ("子群体判别", test_discriminator),
        ("基础系统", test_basic_system),
        ("数据导出", test_data_export)
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ 测试 '{name}' 发生异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{status}: {name}")
    
    print("\n" + "-" * 60)
    print(f"总计: {passed}/{total} 测试通过 ({passed/total*100:.1f}%)")
    print("-" * 60)
    
    if passed == total:
        print("\n🎉 所有测试通过! 系统运行正常!")
        return True
    else:
        print(f"\n⚠️  有 {total - passed} 个测试失败，请检查错误信息")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='配额系统测试')
    parser.add_argument('--test', type=str, default='all',
                       choices=['all', 'imports', 'quota', 'annealing', 
                               'discriminator', 'system', 'export'],
                       help='要运行的测试')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        success = run_all_tests()
    elif args.test == 'imports':
        success = test_imports()
    elif args.test == 'quota':
        success = test_quota_creation()
    elif args.test == 'annealing':
        success = test_annealing()
    elif args.test == 'discriminator':
        success = test_discriminator()
    elif args.test == 'system':
        success = test_basic_system()
    elif args.test == 'export':
        success = test_data_export()
    
    sys.exit(0 if success else 1)

