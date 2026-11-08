import numpy as np
import pandas as pd
import re
import glob
from pathlib import Path

# =========================
# 工具函数：非线性外推组件（动态levels版本）
# =========================
def _clip01(x):
    return float(np.clip(x, 0.0, 1.0))

def poly_project(years, values, target_year, deg=2):
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    # 点数不足时降阶或退化
    uniq = np.unique(years)
    if len(uniq) == 1:
        return float(values.mean())
    deg_eff = min(deg, len(uniq) - 1)  # 多项式次数不能超过点数-1
    if deg_eff <= 1:
        # 退化为线性
        A = np.vstack([years, np.ones_like(years)]).T
        a, b = np.linalg.lstsq(A, values, rcond=None)[0]
        return _clip01(a * target_year + b)
    # 标准化年份以降低病态
    y0 = years.mean()
    ys = years - y0
    coeff = np.polyfit(ys, values, deg=deg_eff)
    pred = np.polyval(coeff, target_year - y0)
    return _clip01(pred)

def exp_project(years, values, target_year):
    # 指数趋势：p(t) ≈ c * exp(k * t)
    # 需 p>0，先做下界平滑
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    eps = 1e-6
    v = np.clip(values, eps, 1 - eps)
    if len(np.unique(years)) < 2:
        return float(values.mean())
    # 对数线性回归：log p = k t + log c
    A = np.vstack([years, np.ones_like(years)]).T
    k, logc = np.linalg.lstsq(A, np.log(v), rcond=None)[0]
    pred = np.exp(k * target_year + logc)
    return _clip01(pred)

def loess_project(years, values, target_year, frac=0.8):
    """
    轻量 LOESS（本地加权线性回归）的简化实现：
    - 对目标点进行一次加权线性拟合，权重按 tri-cube kernel 衰减
    - 适合点数很少的平滑趋势
    """
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    n = len(years)
    if len(np.unique(years)) < 2 or n < 3:
        # 点太少时退化为线性
        A = np.vstack([years, np.ones_like(years)]).T
        a, b = np.linalg.lstsq(A, values, rcond=None)[0]
        return _clip01(a * target_year + b)

    # 计算距离与带宽
    d = np.abs(years - target_year)
    # 选择带宽为 frac 分位数的最大距离
    h = np.quantile(d, min(0.999, max(0.1, frac)))
    h = max(h, 1e-6)
    # tri-cube 权重
    w = (1 - (d / h) ** 3) ** 3
    w[d > h] = 0.0
    # 带权线性拟合
    W = np.diag(w)
    X = np.vstack([years, np.ones_like(years)]).T
    try:
        beta = np.linalg.lstsq(W @ X, W @ values, rcond=None)[0]
        pred = beta[0] * target_year + beta[1]
    except np.linalg.LinAlgError:
        pred = float(values.mean())
    return _clip01(pred)

def linear_project(years, values, target_year):
    # 保留原来的线性以便回退
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    if len(np.unique(years)) < 2:
        return float(values.mean())
    A = np.vstack([years, np.ones_like(years)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    pred = a * target_year + b
    return _clip01(pred)

def nonlinear_project(
    years,
    values,
    target_year,
    method="auto",
    poly_deg=2,
    use_loess=True,
    use_exp=True,
):
    """
    返回更拟合但稳健的预测值。
    method:
      - "poly": 使用多项式（默认二次）
      - "loess": 局部回归
      - "exp": 指数趋势
      - "linear": 线性（回退）
      - "auto": 多模型集成（默认）
    """
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)

    # 少点数保护：直接线性或均值
    if len(np.unique(years)) < 2:
        return float(values.mean())

    if method == "linear":
        return linear_project(years, values, target_year)
    if method == "poly":
        return poly_project(years, values, target_year, deg=poly_deg)
    if method == "loess":
        return loess_project(years, values, target_year)
    if method == "exp":
        return exp_project(years, values, target_year)

    # auto: 集成多个弱模型，提升稳健性
    preds = []
    weights = []

    # 多项式（默认二次）
    p_poly = poly_project(years, values, target_year, deg=poly_deg)
    preds.append(p_poly); weights.append(1.0)

    # LOESS（若启用）
    if use_loess:
        p_loess = loess_project(years, values, target_year, frac=0.8)
        preds.append(p_loess); weights.append(1.0)

    # 指数趋势（若启用）
    if use_exp:
        p_exp = exp_project(years, values, target_year)
        preds.append(p_exp); weights.append(0.6)  # 稍低权重，避免外推爆炸

    # 线性作为锚点
    p_lin = linear_project(years, values, target_year)
    preds.append(p_lin); weights.append(0.8)

    preds = np.array(preds, dtype=float)
    weights = np.array(weights, dtype=float)
    pred = float(np.average(preds, weights=weights))
    return _clip01(pred)

# =========================
# 2) 外推到目标年（非线性版本）：边际 + 选定一张两两交叉（默认 B-C）
# 支持动态levels
# =========================
def project_to_year_nonlinear(
    stats_by_year,
    target_year,
    attr_names=None,
    levels=None,
    method="auto",
    poly_deg=2,
    use_loess=True,
    use_exp=True
):
    """
    非线性外推到目标年份（动态levels版本）
    
    Args:
        stats_by_year: 年度统计数据字典
        target_year: 目标年份
        attr_names: 属性名称列表 [attr_a, attr_b, attr_c]，如果为None则使用全局ATTR_NAMES
        levels: levels列表，如果为None则使用全局levels
        method: 预测方法 ("auto", "poly", "loess", "exp", "linear")
        poly_deg: 多项式度数
        use_loess: 是否使用LOESS
        use_exp: 是否使用指数趋势
    
    Returns:
        tuple: (P_a, P_b, P_c, P_bc)
    """
    # 导入必要的函数和变量
    try:
        from get_distribution_dynamic import ATTR_A, ATTR_B, ATTR_C, normalize, get_levels
        # 如果没有指定levels，使用动态levels
        if levels is None:
            levels = get_levels()
    except ImportError:
        # 如果无法导入，使用传统的导入方式
        try:
            from get_distribution import ATTR_A, ATTR_B, ATTR_C, normalize
            if levels is None:
                levels = ['1', '2', '3']  # 默认值
        except ImportError:
            # 完全无法导入时，使用局部定义
            def normalize(p):
                s = float(p.sum())
                return p / s if s > 0 else p
            
            if attr_names is None:
                attr_a, attr_b, attr_c = "A", "B", "C"
            else:
                attr_a, attr_b, attr_c = attr_names
            
            if levels is None:
                levels = ['1', '2', '3']
    
    # 如果没有指定attr_names，使用全局ATTR_A, ATTR_B, ATTR_C
    if attr_names is None:
        try:
            attr_a, attr_b, attr_c = ATTR_A, ATTR_B, ATTR_C
        except NameError:
            attr_a, attr_b, attr_c = "A", "B", "C"
    else:
        attr_a, attr_b, attr_c = attr_names
    
    years = sorted(stats_by_year.keys())
    
    # 初始化概率序列（使用动态levels）
    P_a = pd.Series(index=levels, dtype=float)
    P_b = pd.Series(index=levels, dtype=float)
    P_c = pd.Series(index=levels, dtype=float)

    # 边际外推（对每个水平）
    for lvl in levels:
        # 提取历史数据序列
        a_values = [stats_by_year[y][f"P_{attr_a}"][lvl] for y in years]
        b_values = [stats_by_year[y][f"P_{attr_b}"][lvl] for y in years]
        c_values = [stats_by_year[y][f"P_{attr_c}"][lvl] for y in years]
        
        # 非线性外推
        P_a[lvl] = nonlinear_project(
            years, a_values, target_year,
            method=method, poly_deg=poly_deg, use_loess=use_loess, use_exp=use_exp
        )
        P_b[lvl] = nonlinear_project(
            years, b_values, target_year,
            method=method, poly_deg=poly_deg, use_loess=use_loess, use_exp=use_exp
        )
        P_c[lvl] = nonlinear_project(
            years, c_values, target_year,
            method=method, poly_deg=poly_deg, use_loess=use_loess, use_exp=use_exp
        )
    
    # 归一化边际分布
    P_a, P_b, P_c = normalize(P_a), normalize(P_b), normalize(P_c)

    # 选用 B-C 二阶交叉承载相关性（动态维度）
    n_levels = len(levels)
    bc_frames = np.vstack([stats_by_year[y][f"P_{attr_b}_{attr_c}"].values.flatten() for y in years])
    pred_bc = []
    for cell in range(n_levels * n_levels):
        cell_values = bc_frames[:, cell]
        pred_bc.append(
            nonlinear_project(
                years, cell_values, target_year,
                method=method, poly_deg=poly_deg, use_loess=use_loess, use_exp=use_exp
            )
        )
    
    # 归一化联合分布
    pred_bc = np.array(pred_bc, dtype=float)
    s = pred_bc.sum()
    if s > 0:
        pred_bc = pred_bc / s
    
    P_bc = pd.DataFrame(pred_bc.reshape(n_levels, n_levels), index=levels, columns=levels)
    return P_a, P_b, P_c, P_bc

# =========================
# 便捷函数：支持不同levels范围的预测
# =========================
def predict_with_dynamic_levels(
    stats_by_year,
    target_year,
    levels_config,
    attr_names=None,
    method="auto"
):
    """
    使用动态levels配置进行预测的便捷函数
    
    Args:
        stats_by_year: 年度统计数据字典
        target_year: 目标年份
        levels_config: levels配置，可以是：
                      - 列表，如 ['1','2','3','4','5']
                      - 元组(start, end)，如 (1, 10)
                      - 字符串范围，如 "1-5"
        attr_names: 属性名称列表
        method: 预测方法
    
    Returns:
        tuple: (P_a, P_b, P_c, P_bc, levels_used)
    """
    # 处理levels_config
    if isinstance(levels_config, str) and '-' in levels_config:
        # 处理"1-5"格式
        start, end = map(int, levels_config.split('-'))
        levels = [str(i) for i in range(start, end + 1)]
    elif isinstance(levels_config, (tuple, list)) and len(levels_config) == 2 and all(isinstance(x, int) for x in levels_config):
        # 处理(1, 5)格式
        start, end = levels_config
        levels = [str(i) for i in range(start, end + 1)]
    else:
        # 直接使用列表格式
        levels = [str(level) for level in levels_config]
    
    # 调用非线性预测函数
    P_a, P_b, P_c, P_bc = project_to_year_nonlinear(
        stats_by_year=stats_by_year,
        target_year=target_year,
        attr_names=attr_names,
        levels=levels,
        method=method
    )
    
    return P_a, P_b, P_c, P_bc, levels

# =========================
# 演示函数
# =========================
def demo_dynamic_levels():
    """
    演示动态levels功能的函数
    """
    print("=" * 60)
    print("动态Levels非线性外推演示")
    print("=" * 60)
    
    # 模拟一些简单的历史数据
    years = [2010, 2015, 2020]
    
    # 为不同的levels配置创建模拟数据
    levels_configs = [
        ['1', '2', '3'],
        ['1', '2', '3', '4', '5'],
        (1, 7),
        "1-4"
    ]
    
    for i, levels_config in enumerate(levels_configs):
        print(f"\n--- 配置 {i+1}: {levels_config} ---")
        
        # 处理levels配置
        if isinstance(levels_config, str) and '-' in levels_config:
            start, end = map(int, levels_config.split('-'))
            levels = [str(i) for i in range(start, end + 1)]
        elif isinstance(levels_config, (tuple, list)) and len(levels_config) == 2 and all(isinstance(x, int) for x in levels_config):
            start, end = levels_config
            levels = [str(i) for i in range(start, end + 1)]
        else:
            levels = [str(level) for level in levels_config]
        
        print(f"使用levels: {levels}")
        
        # 创建模拟的历史数据
        np.random.seed(42)  # 确保可复现
        n_levels = len(levels)
        
        # 模拟年度统计数据
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
                'P_A': pd.Series(pa_data, index=levels),
                'P_B': pd.Series(pb_data, index=levels),
                'P_C': pd.Series(pc_data, index=levels),
                'P_B_C': pd.DataFrame(bc_data, index=levels, columns=levels)
            }
        
        # 进行预测
        try:
            P_a, P_b, P_c, P_bc = project_to_year_nonlinear(
                stats_by_year=stats_by_year,
                target_year=2025,
                attr_names=['A', 'B', 'C'],
                levels=levels,
                method="auto"
            )
            
            print(f"预测成功! 边际分布维度: {len(P_a)}")
            print(f"联合分布维度: {P_bc.shape}")
            print(f"P_A预测: {P_a.round(3).to_dict()}")
            
        except Exception as e:
            print(f"预测失败: {e}")

if __name__ == "__main__":
    demo_dynamic_levels()
