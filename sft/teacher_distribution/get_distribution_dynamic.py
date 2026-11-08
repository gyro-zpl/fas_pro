import numpy as np
import pandas as pd
import re
import glob
from pathlib import Path
from get_distribution_nonlinear_dynamic import project_to_year_nonlinear

# =========================
# 配置：你的数据目录、年份与变量名
# =========================
DATA_DIR = "/home/cyyuan/FAS_PRO/data/gss"
TARGET_YEARS = {2004, 2008, 2012, 2014, 2016, 2018, 2021, 2022}

# 默认属性名称，可以通过set_attr_names函数修改
ATTR_NAMES = ["nateduc", "natdrug", "natcrime"]
ATTR_A, ATTR_B, ATTR_C = ATTR_NAMES
TARGET_YEAR = 2024
SEED = 2024  # 采样可复现用

# 动态levels配置 - 可以通过set_levels函数修改
levels = ['1', '2', '3']  # 默认值，可以动态修改

# 全局变量初始化
df_all = None
stats_by_year = None
P_a_0 = P_b_0 = P_c_0 = P_bc_0 = None
P_b_adj = None
Q_joint = None
Pa_pred = Pb_pred = Pc_pred = None
predicted_joints = None
samples_20 = None

# =========================
# 动态levels配置函数
# =========================
def set_levels(new_levels):
    """
    设置新的levels范围
    
    Args:
        new_levels: 新的levels列表，如 ['1','2','3'] 或 ['1','2','3','4','5'] 等
    """
    global levels
    levels = [str(level) for level in new_levels]  # 确保都是字符串
    print(f"Levels已更新为: {levels}")
    return levels

def get_levels():
    """
    获取当前的levels配置
    
    Returns:
        list: 当前的levels列表
    """
    return levels

def generate_levels_range(start=1, end=10):
    """
    生成从start到end的levels范围
    
    Args:
        start: 起始值（包含）
        end: 结束值（包含）
    
    Returns:
        list: levels列表
    """
    return [str(i) for i in range(start, end + 1)]

# =========================
# 属性名称配置函数
# =========================
def set_attr_names(attr_a, attr_b, attr_c):
    """
    设置三个属性的名称
    
    Args:
        attr_a: 第一个属性名称（对应A）
        attr_b: 第二个属性名称（对应B）
        attr_c: 第三个属性名称（对应C）
    """
    global ATTR_NAMES, ATTR_A, ATTR_B, ATTR_C
    ATTR_NAMES = [attr_a, attr_b, attr_c]
    ATTR_A, ATTR_B, ATTR_C = ATTR_NAMES
    print(f"属性名称已更新为: A={ATTR_A}, B={ATTR_B}, C={ATTR_C}")

def get_attr_names():
    """
    获取当前的属性名称
    
    Returns:
        tuple: (ATTR_A, ATTR_B, ATTR_C)
    """
    return ATTR_A, ATTR_B, ATTR_C

# =========================
# 主要处理函数
# =========================
def process_data_with_attrs(attr_names=None, levels_config=None, data_dir=None, target_years=None, target_year=None):
    """
    使用指定的属性名称和levels配置处理数据
    
    Args:
        attr_names: 三个属性名称的列表，如 ["natcrime", "natdrug", "nateduc"]
        levels_config: levels配置，可以是列表或元组(start, end)
        data_dir: 数据目录路径
        target_years: 目标年份集合
        target_year: 预测目标年份
    
    Returns:
        tuple: (Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints)
    """
    global ATTR_NAMES, ATTR_A, ATTR_B, ATTR_C, DATA_DIR, TARGET_YEARS, TARGET_YEAR, levels
    
    # 如果提供了参数，则更新全局配置
    if attr_names is not None:
        set_attr_names(*attr_names)
    
    if levels_config is not None:
        if isinstance(levels_config, (list, tuple)) and len(levels_config) == 2 and all(isinstance(x, int) for x in levels_config):
            # 如果是(start, end)元组，生成范围
            set_levels(generate_levels_range(levels_config[0], levels_config[1]))
        else:
            # 否则直接设置
            set_levels(levels_config)
    
    if data_dir is not None:
        DATA_DIR = data_dir
    if target_years is not None:
        TARGET_YEARS = target_years
    if target_year is not None:
        TARGET_YEAR = target_year
    
    # 执行数据处理
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    P_a_0, P_b_0, P_c_0, P_bc_0 = project_to_year_nonlinear(stats_by_year, TARGET_YEAR)
    P_b_adj = inject_scenario(P_b_0, delta_3=0.00)
    Q_joint = ipf_joint(P_a_0, P_b_adj, P_c_0, P_bc_0, iters=300)
    
    Pa_pred, Pb_pred, Pc_pred = marginal_from_Q(Q_joint)
    predicted_joints = extract_pairwise_joints(Q_joint)
    
    return Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints

# =========================
# 读取多CSV并按年份合并（从文件名提取年份）
# =========================
def load_and_process_data():
    """加载和处理CSV数据"""
    YEAR_REGEX = re.compile(r"(\d{4})(?:\.\w+)?$")

    def extract_year_from_filename(path: str):
        name = Path(path).name
        m = YEAR_REGEX.search(name)
        return int(m.group(1)) if m else None

    all_csv = sorted(glob.glob(f"{DATA_DIR}/*.csv"))
    picked = []
    for p in all_csv:
        y = extract_year_from_filename(p)
        if y is None:
            continue
        if y in TARGET_YEARS:
            picked.append((y, p))

    if not picked:
        raise RuntimeError(f"在 {DATA_DIR} 下未找到匹配年份 {sorted(TARGET_YEARS)} 的CSV。")

    print("将读取的文件：")
    for y, p in picked:
        print(f"- {y}: {p}")

    frames = []
    for y, p in picked:
        dtype_dict = {attr: str for attr in ATTR_NAMES}
        df = pd.read_csv(p, dtype=dtype_dict)
        df["year"] = y
        needed_cols = ["year"] + ATTR_NAMES
        missing = [c for c in needed_cols if c not in df.columns]
        if missing:
            raise ValueError(f"文件 {p} 缺少列：{missing}。请检查列名或在此处做列名映射。")
        for col in ATTR_NAMES:
            df[col] = df[col].astype(str).str.strip()
        # 过滤取值域 - 使用动态levels
        mask = pd.Series(True, index=df.index)
        for col in ATTR_NAMES:
            mask &= df[col].isin(levels)
        df = df.loc[mask, needed_cols].copy()
        frames.append(df)

    df_all = pd.concat(frames, ignore_index=True)
    print("合并完成，样本量：", len(df_all))
    print("年份分布：")
    print(df_all["year"].value_counts().sort_index())
    
    return df_all

# =========================
# 工具函数
# =========================
def normalize(p: pd.Series) -> pd.Series:
    s = float(p.sum())
    return p / s if s > 0 else p

def linear_project(years, values, target_year):
    years = np.array(years, dtype=float)
    values = np.array(values, dtype=float)
    if len(np.unique(years)) < 2:
        return float(values.mean())
    A = np.vstack([years, np.ones_like(years)]).T
    a, b = np.linalg.lstsq(A, values, rcond=None)[0]
    pred = a * target_year + b
    return float(np.clip(pred, 0.0, 1.0))

# =========================
# 1) 年度边际与两两联合分布（动态变量名）
# =========================
def year_stats(df):
    res = {}
    for y, d in df.groupby("year"):
        pa = d[ATTR_A].value_counts().reindex(levels, fill_value=0).astype(float)
        pb = d[ATTR_B].value_counts().reindex(levels, fill_value=0).astype(float)
        pc = d[ATTR_C].value_counts().reindex(levels, fill_value=0).astype(float)
        pa, pb, pc = normalize(pa), normalize(pb), normalize(pc)

        # 两两交叉 - 使用动态levels
        ct_b_c = pd.crosstab(d[ATTR_B], d[ATTR_C]).reindex(index=levels, columns=levels, fill_value=0).astype(float)
        ct_b_c = ct_b_c / ct_b_c.values.sum() if ct_b_c.values.sum() > 0 else ct_b_c

        ct_a_b = pd.crosstab(d[ATTR_A], d[ATTR_B]).reindex(index=levels, columns=levels, fill_value=0).astype(float)
        ct_a_b = ct_a_b / ct_a_b.values.sum() if ct_a_b.values.sum() > 0 else ct_a_b

        ct_a_c = pd.crosstab(d[ATTR_A], d[ATTR_C]).reindex(index=levels, columns=levels, fill_value=0).astype(float)
        ct_a_c = ct_a_c / ct_a_c.values.sum() if ct_a_c.values.sum() > 0 else ct_a_c

        res[y] = {
            f"P_{ATTR_A}": pa,
            f"P_{ATTR_B}": pb,
            f"P_{ATTR_C}": pc,
            f"P_{ATTR_B}_{ATTR_C}": ct_b_c,
            f"P_{ATTR_A}_{ATTR_B}": ct_a_b,
            f"P_{ATTR_A}_{ATTR_C}": ct_a_c
        }
    return res

# =========================
# 2) 外推到目标年：边际 + 选定一张两两交叉（默认 B-C）
# =========================
def project_to_year_linear(stats_by_year, target_year):
    years = sorted(stats_by_year.keys())
    P_a = pd.Series(index=levels, dtype=float)
    P_b = pd.Series(index=levels, dtype=float)
    P_c = pd.Series(index=levels, dtype=float)
    for lvl in levels:
        P_a[lvl] = linear_project(years, [stats_by_year[y][f"P_{ATTR_A}"][lvl] for y in years], target_year)
        P_b[lvl] = linear_project(years, [stats_by_year[y][f"P_{ATTR_B}"][lvl] for y in years], target_year)
        P_c[lvl] = linear_project(years, [stats_by_year[y][f"P_{ATTR_C}"][lvl] for y in years], target_year)
    P_a, P_b, P_c = normalize(P_a), normalize(P_b), normalize(P_c)

    # 选用 B-C 二阶交叉来承载相关性 - 动态维度
    n_levels = len(levels)
    bc_frames = np.vstack([stats_by_year[y][f"P_{ATTR_B}_{ATTR_C}"].values.flatten() for y in years])
    pred_bc = []
    for cell in range(n_levels * n_levels):
        pred_bc.append(linear_project(years, bc_frames[:, cell], target_year))
    pred_bc = np.array(pred_bc)
    pred_bc = pred_bc / pred_bc.sum() if pred_bc.sum() > 0 else pred_bc
    P_bc = pd.DataFrame(pred_bc.reshape(n_levels, n_levels), index=levels, columns=levels)

    return P_a, P_b, P_c, P_bc

# =========================
# 3) （可选）情景先验注入：提升某变量的最高等级
# =========================
def inject_scenario(P_attr: pd.Series, delta_high=0.0, high_level=None):
    """
    情景注入函数，支持动态levels
    
    Args:
        P_attr: 属性的概率分布
        delta_high: 对最高等级的增量调整
        high_level: 指定的高等级，如果为None则使用levels中的最后一个
    """
    P = P_attr.copy()
    if high_level is None:
        high_level = levels[-1]  # 默认使用最后一个level作为高等级
    
    if delta_high != 0.0:
        if delta_high > 0:
            inc = min(delta_high, 1.0 - P[high_level])
        else:
            inc = max(delta_high, -P[high_level])
        
        # 计算其他等级的总和
        other_levels = [lvl for lvl in levels if lvl != high_level]
        remainder = sum(P[lvl] for lvl in other_levels)
        
        if remainder > 0 and inc != 0.0:
            P[high_level] += inc
            # 按比例减少其他等级
            for lvl in other_levels:
                P[lvl] -= inc * (P[lvl] / remainder)
    
    P = P.clip(lower=0.0)
    return normalize(P)

# =========================
# 4) IPF 融合：构造动态维度的联合分布 Q(A,B,C)
# =========================
def ipf_joint(Pa, Pb, Pc, Pbc, iters=300, eps=1e-12):
    """
    支持动态levels的IPF函数
    """
    n_levels = len(levels)
    
    # 初始独立分布
    Q = np.zeros((n_levels, n_levels, n_levels), dtype=float)
    for i, a in enumerate(levels):
        for j, b in enumerate(levels):
            for k, c in enumerate(levels):
                Q[i, j, k] = Pa[a] * Pb[b] * Pc[c]
    Q /= Q.sum()

    Pbc_arr = Pbc.values  # (n_levels, n_levels)

    for _ in range(iters):
        # 匹配 sum_A Q = Pbc
        sum_over_a = Q.sum(axis=0) + eps  # (B,C)
        Q *= (Pbc_arr / sum_over_a)[None, :, :]

        # 匹配 sum_{B,C} Q = Pa
        sum_over_bc = Q.sum(axis=(1, 2)) + eps  # (A,)
        for i in range(n_levels):
            Q[i, :, :] *= (Pa[levels[i]] / sum_over_bc[i])

        # 归一化
        s = Q.sum()
        if s > 0:
            Q /= s

    # 轻微平滑
    alpha = 1e-6
    Q = (Q + alpha) / (Q + alpha).sum()
    return Q

# =========================
# 5) 采样与派生两两联合
# =========================
def sample_from_joint(Q, n=20, seed=None):
    """
    从联合分布中采样，支持动态levels
    """
    rng = np.random.default_rng(seed)
    flat = (Q / Q.sum()).flatten()
    idx = rng.choice(len(flat), size=n, replace=True, p=flat)
    samples = []
    n_levels = len(levels)
    
    for m in idx:
        i = m // (n_levels * n_levels)           # A index
        j = (m % (n_levels * n_levels)) // n_levels     # B index
        k = m % n_levels                         # C index
        samples.append([levels[i], levels[j], levels[k]])
    return samples

def extract_pairwise_joints(Q_joint):
    """
    提取成对联合分布，支持动态levels
    """
    # P(B,C) = sum_A Q
    P_b_c = Q_joint.sum(axis=0)
    # P(A,B) = sum_C Q
    P_a_b = Q_joint.sum(axis=2)
    # P(A,C) = sum_B Q
    P_a_c = Q_joint.sum(axis=1)

    return {
        f"P_{ATTR_B}_{ATTR_C}": pd.DataFrame(P_b_c, index=levels, columns=levels),
        f"P_{ATTR_A}_{ATTR_B}": pd.DataFrame(P_a_b, index=levels, columns=levels),
        f"P_{ATTR_A}_{ATTR_C}": pd.DataFrame(P_a_c, index=levels, columns=levels),
    }

# =========================
# 6) 打印结果（动态变量名）
# =========================
def marginal_from_Q(Q):
    """
    从联合分布中提取边际分布，支持动态levels
    """
    n_levels = len(levels)
    Pa = pd.Series({levels[i]: Q[i, :, :].sum() for i in range(n_levels)})
    Pb = pd.Series({levels[j]: Q[:, j, :].sum() for j in range(n_levels)})
    Pc = pd.Series({levels[k]: Q[:, :, k].sum() for k in range(n_levels)})
    return Pa, Pb, Pc

def print_results():
    """
    打印预测结果，支持动态levels
    """
    print(f"\n=== 预测的 {TARGET_YEAR} 年分布 ===")
    print("边际分布:")
    print(f"P({ATTR_A}):")
    print(Pa_pred.round(3))
    print(f"P({ATTR_B}):")
    print(Pb_pred.round(3))
    print(f"P({ATTR_C}):")
    print(Pc_pred.round(3))

    print("\n联合分布:")
    print(f"P({ATTR_B}, {ATTR_C}):")
    print(predicted_joints[f"P_{ATTR_B}_{ATTR_C}"].round(3))
    print(f"P({ATTR_A}, {ATTR_B}):")
    print(predicted_joints[f"P_{ATTR_A}_{ATTR_B}"].round(3))
    print(f"P({ATTR_A}, {ATTR_C}):")
    print(predicted_joints[f"P_{ATTR_A}_{ATTR_C}"].round(3))

    print(f"\nSampled {len(samples_20)} rows ([{ATTR_A!r},{ATTR_B!r},{ATTR_C!r}]):")
    for row in samples_20[:5]:  # 只显示前5行
        print(str(row))

# =========================
# 便捷函数：获取当前配置的分布
# =========================
def get_current_distributions():
    """
    获取当前配置下的所有分布
    
    Returns:
        dict: 包含所有分布的字典
    """
    global Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints
    
    if Q_joint is None:
        # 如果还没有处理数据，则处理一次
        Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints = process_data_with_attrs()
    
    return {
        'Q_joint': Q_joint,
        'Pa_pred': Pa_pred,
        'Pb_pred': Pb_pred, 
        'Pc_pred': Pc_pred,
        'predicted_joints': predicted_joints,
        'levels': levels.copy(),
        'attr_names': get_attr_names()
    }

def get_distributions_for_attrs_and_levels(attr_a, attr_b, attr_c, levels_config):
    """
    为指定的属性名称和levels配置获取分布
    
    Args:
        attr_a: 第一个属性名称
        attr_b: 第二个属性名称  
        attr_c: 第三个属性名称
        levels_config: levels配置，可以是列表或元组(start, end)
        
    Returns:
        tuple: (Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints)
    """
    return process_data_with_attrs(attr_names=[attr_a, attr_b, attr_c], levels_config=levels_config)

# 当直接运行脚本时，使用默认配置处理数据
if __name__ == "__main__":
    print("=" * 80)
    print("📊 GSS数据分析与预测系统 (动态Levels版本)")
    print("=" * 80)
    
    # 显示配置信息
    print(f"🔧 当前配置:")
    print(f"   数据目录: {DATA_DIR}")
    print(f"   目标年份集合: {sorted(TARGET_YEARS)}")
    print(f"   预测目标年份: {TARGET_YEAR}")
    print(f"   分析变量: {ATTR_NAMES}")
    print(f"   变量别名: A={ATTR_A}, B={ATTR_B}, C={ATTR_C}")
    print(f"   分析等级: {levels} (支持动态配置)")
    
    # 演示动态levels功能
    print(f"\n🎯 演示动态levels功能:")
    print(f"   当前levels: {levels}")
    
    # 可以测试不同的levels配置
    # 示例1: 使用1-5的范围
    # set_levels(generate_levels_range(1, 5))
    # print(f"   设置为1-5范围: {levels}")
    
    # 示例2: 使用自定义列表
    # set_levels(['低', '中', '高'])
    # print(f"   设置为自定义标签: {levels}")
    
    # 恢复默认设置进行演示
    set_levels(['1', '2', '3'])
    
    # 加载和处理数据
    print(f"\n🔄 正在加载历史数据...")
    df_all = load_and_process_data()
    stats_by_year = year_stats(df_all)
    
    historical_years = sorted(stats_by_year.keys())
    print(f"✅ 成功加载 {len(historical_years)} 个年份的数据: {historical_years}")
    
    # 进行预测
    print(f"\n🎯 正在进行 {TARGET_YEAR} 年预测...")
    P_a_0, P_b_0, P_c_0, P_bc_0 = project_to_year_nonlinear(stats_by_year, TARGET_YEAR)
    P_b_adj = inject_scenario(P_b_0, delta_high=0.00)
    Q_joint = ipf_joint(P_a_0, P_b_adj, P_c_0, P_bc_0, iters=300)
    Pa_pred, Pb_pred, Pc_pred = marginal_from_Q(Q_joint)
    
    # 输出预测结果
    print(f"\n📈 {TARGET_YEAR}年预测结果:")
    print("-" * 50)
    
    for var_idx, (var_name, pred_dist) in enumerate([(ATTR_A, Pa_pred), (ATTR_B, Pb_pred), (ATTR_C, Pc_pred)]):
        print(f"\n{var_name.upper()} ({['A', 'B', 'C'][var_idx]}) 预测分布:")
        for level in levels:
            prob = pred_dist[level]
            print(f"  Level {level}: {prob:.4f} ({prob*100:.2f}%)")
    
    # 显示历史联合分布（前两年）
    print(f"\n=== 历史年份的联合分布 (前两年) ===")
    for year in sorted(stats_by_year.keys())[:2]:
        print(f"\n--- {year}年 ---")
        print(f"P({ATTR_B}, {ATTR_C}):")
        print(stats_by_year[year][f"P_{ATTR_B}_{ATTR_C}"].round(3))
    
    # 显示预测联合分布
    print(f"\n=== {TARGET_YEAR}年预测联合分布 ===")
    print(f"P({ATTR_B}, {ATTR_C}) - 预测:")
    print(P_bc_0.round(4))
    
    # 总结信息
    print(f"\n" + "=" * 80)
    print(f"✅ 预测完成! 基于 {len(historical_years)} 个历史年份数据预测 {TARGET_YEAR} 年")
    print(f"📊 使用非线性外推方法，结合IPF迭代比例拟合")
    print(f"🎯 支持动态levels配置，当前levels: {levels}")
    print(f"🔧 可调用以下函数进行配置:")
    print(f"   - set_levels(['1','2','3','4','5']) 或 set_levels(generate_levels_range(1,10))")
    print(f"   - process_data_with_attrs(levels_config=(1,10)) 或 levels_config=['低','中','高']")
    print(f"=" * 80)
else:
    # 当作为模块导入时，不立即处理数据（变量已在顶部初始化）
    pass
