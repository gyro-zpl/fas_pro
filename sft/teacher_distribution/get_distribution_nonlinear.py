import numpy as np
import pandas as pd
import re
import glob
from pathlib import Path
# =========================
# 工具函数：非线性外推组件
# =========================
levels = ['1', '2', '3']
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
# =========================
def project_to_year_nonlinear(
    stats_by_year,
    target_year,
    attr_names=None,
    method="auto",
    poly_deg=2,
    use_loess=True,
    use_exp=True
):
    """
    非线性外推到目标年份
    
    Args:
        stats_by_year: 年度统计数据字典
        target_year: 目标年份
        attr_names: 属性名称列表 [attr_a, attr_b, attr_c]，如果为None则使用全局ATTR_NAMES
        method: 预测方法 ("auto", "poly", "loess", "exp", "linear")
        poly_deg: 多项式度数
        use_loess: 是否使用LOESS
        use_exp: 是否使用指数趋势
    
    Returns:
        tuple: (P_a, P_b, P_c, P_bc)
    """
    # 导入必要的函数和变量
    from get_distribution import ATTR_A, ATTR_B, ATTR_C, normalize
    
    # 如果没有指定attr_names，使用全局ATTR_A, ATTR_B, ATTR_C
    if attr_names is None:
        attr_a, attr_b, attr_c = ATTR_A, ATTR_B, ATTR_C
    else:
        attr_a, attr_b, attr_c = attr_names
    years = sorted(stats_by_year.keys())
    
    # 初始化概率序列
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

    # 选用 B-C 二阶交叉承载相关性
    bc_frames = np.vstack([stats_by_year[y][f"P_{attr_b}_{attr_c}"].values.flatten() for y in years])
    pred_bc = []
    for cell in range(9):
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
    
    P_bc = pd.DataFrame(pred_bc.reshape(3, 3), index=levels, columns=levels)
    return P_a, P_b, P_c, P_bc