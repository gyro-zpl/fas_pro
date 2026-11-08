import pandas as pd
import numpy as np
from itertools import product
from scipy.spatial.distance import jensenshannon
from scipy.stats import chisquare, entropy
try:
    from utils import COLUMNS
except ImportError:
    # 如果无法导入，使用默认值
    COLUMNS = ["nateduc", "natdrug", "natcrime"]


def unify_probs(series, value_union=None, eps=1e-9):
    # 自动发现类别；也可外部传入统一类别集 value_union
    values = list(value_union) if value_union is not None else sorted(series.dropna().unique().tolist())
    counts = series.value_counts().reindex(values, fill_value=0).astype(float)
    p = (counts.values + eps) / (counts.sum() + eps * len(values))
    return p, values

def js_divergence(p, q):
    return float(jensenshannon(p, q) ** 2)  # JS distance^2 = JS divergence

def kl_divergence(p, q, eps=1e-12):
    """
    计算KL散度 KL(P||Q) = sum(P * log(P/Q))
    
    Args:
        p: 真实分布 (参考分布)
        q: 生成分布 (目标分布)
        eps: 平滑参数，避免log(0)
    
    Returns:
        float: KL散度值
    """
    # 确保分布归一化
    p = np.array(p)
    q = np.array(q)
    p = p / p.sum()
    q = q / q.sum()
    
    # 添加平滑避免0值
    p = p + eps
    q = q + eps
    p = p / p.sum()
    q = q / q.sum()
    
    # 使用scipy.stats.entropy计算KL散度
    return float(entropy(p, q))

def symmetric_kl_divergence(p, q, eps=1e-12):
    """
    计算对称KL散度 (KL(P||Q) + KL(Q||P)) / 2
    
    Args:
        p: 真实分布
        q: 生成分布
        eps: 平滑参数
    
    Returns:
        float: 对称KL散度值
    """
    return (kl_divergence(p, q, eps) + kl_divergence(q, p, eps)) / 2

def interpret_kl_directions(kl_real_gen, kl_gen_real):
    """
    解释KL散度两个方向的含义并给出建议
    
    Args:
        kl_real_gen: KL(real||gen) - 模式遗漏程度
        kl_gen_real: KL(gen||real) - 虚假模式程度
    
    Returns:
        dict: 包含解释和建议的字典
    """
    ratio = kl_gen_real / kl_real_gen if kl_real_gen > 0 else float('inf')
    
    interpretation = {
        'mode_missing_severity': kl_real_gen,
        'mode_invention_severity': kl_gen_real,
        'ratio_invention_to_missing': ratio,
        'primary_issue': '',
        'recommendation': ''
    }
    
    if ratio > 1.3:
        interpretation['primary_issue'] = '虚假模式问题更突出'
        interpretation['recommendation'] = '重点关注KL(gen||real)，生成模型产生了过多不真实的模式'
    elif ratio < 0.7:
        interpretation['primary_issue'] = '模式遗漏问题更突出'
        interpretation['recommendation'] = '重点关注KL(real||gen)，生成模型遗漏了重要的真实模式'
    else:
        interpretation['primary_issue'] = '模式遗漏和虚假模式相当'
        interpretation['recommendation'] = '建议使用对称KL散度进行综合评估'
    
    return interpretation

def chi_square_from_probs(p_ref, n_gen, p_gen, min_expected=1e-6):
    expected = np.maximum(p_ref * n_gen, min_expected)
    observed = p_gen * n_gen
    stat, pval = chisquare(f_obs=observed, f_exp=expected)
    return float(stat), float(pval)

def merge_rare(p, labels, min_expected, n_gen):
    # 合并期望频数过小的类别到 "OTHER"
    expected = p * n_gen
    keep_mask = expected >= min_expected
    if keep_mask.all():
        return p, labels
    other_mass = p[~keep_mask].sum()
    p_new = np.concatenate([p[keep_mask], [other_mass]])
    labels_new = [labels[i] for i, k in enumerate(keep_mask) if k] + ["OTHER"]
    p_new = p_new / p_new.sum()  # 归一化一次
    return p_new, labels_new

def column_marginal_metrics(real_df, gen_df, col, eps=1e-9, min_expected=5):
    # 统一类别并集
    union_vals = set(real_df[col].dropna().unique()).union(set(gen_df[col].dropna().unique()))
    p_real, labels = unify_probs(real_df[col], union_vals, eps)
    p_gen,  _      = unify_probs(gen_df[col], union_vals, eps)
    
    # JS散度（对样本量不敏感）
    js = js_divergence(p_real, p_gen)
    
    # KL散度计算
    # KL(real||gen): 检测模式遗漏 - 真实模式在生成数据中缺失的程度
    kl_real_gen = kl_divergence(p_real, p_gen, eps)  
    # KL(gen||real): 检测虚假模式 - 生成数据中不真实模式的程度  
    kl_gen_real = kl_divergence(p_gen, p_real, eps)
    # 对称KL散度: 综合评估，平衡两种误差
    kl_symmetric = symmetric_kl_divergence(p_real, p_gen, eps)
    
    # 卡方（对样本量敏感）
    p_real_c, labels_c = merge_rare(p_real, labels, min_expected, len(gen_df))
    # 对齐 gen 的合并
    idx_keep = [i for i, lab in enumerate(labels) if lab in labels_c and lab != "OTHER"]
    p_gen_c = p_gen[idx_keep].copy()
    if "OTHER" in labels_c:
        other_mass = 1.0 - p_gen_c.sum()
        p_gen_c = np.append(p_gen_c, max(other_mass, 0.0))
    p_gen_c = p_gen_c / p_gen_c.sum()
    chi2, pval = chi_square_from_probs(p_real_c, len(gen_df), p_gen_c)
    
    return {
        "JS": js, 
        "KL_real_gen": kl_real_gen,
        "KL_gen_real": kl_gen_real, 
        "KL_symmetric": kl_symmetric,
        "chi2": chi2, 
        "p": pval, 
        "labels": labels, 
        "p_real": p_real.tolist(), 
        "p_gen": p_gen.tolist()
    }

def pair_joint_metrics(real_df, gen_df, col_a, col_b, eps=1e-9):
    """
    计算两两联合分布的JS散度和KL散度
    
    Args:
        real_df: 真实数据
        gen_df: 生成数据
        col_a, col_b: 列名
        eps: 平滑参数
    
    Returns:
        dict: 包含JS、KL散度等指标
    """
    vals_a = set(real_df[col_a].dropna().unique()).union(set(gen_df[col_a].dropna().unique()))
    vals_b = set(real_df[col_b].dropna().unique()).union(set(gen_df[col_b].dropna().unique()))
    vals_a, vals_b = sorted(vals_a), sorted(vals_b)

    def joint_p(df):
        ct = pd.crosstab(df[col_a], df[col_b]).reindex(index=vals_a, columns=vals_b, fill_value=0).astype(float)
        p = ct.values.flatten()
        p = (p + eps) / (p.sum() + eps * (len(vals_a)*len(vals_b)))
        return p

    p_real = joint_p(real_df)
    p_gen  = joint_p(gen_df)
    
    # 计算各种散度指标
    js = js_divergence(p_real, p_gen)
    kl_real_gen = kl_divergence(p_real, p_gen, eps)
    kl_gen_real = kl_divergence(p_gen, p_real, eps)
    kl_symmetric = symmetric_kl_divergence(p_real, p_gen, eps)
    
    return {
        "JS": js,
        "KL_real_gen": kl_real_gen,
        "KL_gen_real": kl_gen_real,
        "KL_symmetric": kl_symmetric
    }

def pair_joint_js(real_df, gen_df, col_a, col_b, eps=1e-9):
    """保持向后兼容性的JS散度函数"""
    return pair_joint_metrics(real_df, gen_df, col_a, col_b, eps)["JS"]

# 连续/大基数数值列（可选）：等频分箱后再比较
def quantile_bin(series, q=10):
    return pd.qcut(series, q=q, duplicates="drop")

# 统一清洗：把字符串/浮点形式（如 '1', '1.0'）转为数值，再转为整数类别
def clean_df(df, columns):
    df = df.copy()
    for c in columns:
        # 去除首尾空白并转为数值
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=columns)
    # 将接近整数的值标准化为整数（避免 1.0/2.0 差异）
    for c in columns:
        df[c] = df[c].round().astype(int)
    return df

# 简单比例统计：按列与整体（跨列合并）
def proportions(df, columns, values=None):
    res = {}
    # 列内比例
    for c in columns:
        if values is None:
            vals = sorted(df[c].dropna().unique().tolist())
        else:
            vals = list(values)
        cnt = df[c].value_counts().reindex(vals, fill_value=0)
        res[c] = (cnt / cnt.sum()).to_dict()
    # 整体比例（把多列视作同一离散域合并统计）
    if values is None:
        vals_all = sorted(pd.unique(pd.concat([df[c] for c in columns], ignore_index=True).dropna()))
    else:
        vals_all = list(values)
    stacked = pd.concat([df[c] for c in columns], ignore_index=True).dropna()
    cnt_all = stacked.value_counts().reindex(vals_all, fill_value=0)
    res["overall"] = (cnt_all / cnt_all.sum()).to_dict()
    return res

# 百分比格式化
def proportions_to_percent(props):
    out = {}
    for col, dist in props.items():
        out[col] = {k: f"{v*100:.2f}%" for k, v in dist.items()}
    return out


def calculate_metrics(COLUMNS,real_path, gen_path):


    real = pd.read_csv(real_path)
    gen  = pd.read_csv(gen_path)

    # 统一清洗与对齐：转数值、去缺失、转整数类别
    real = clean_df(real, COLUMNS)
    gen  = clean_df(gen, COLUMNS)

    print(COLUMNS)
    # 打印样本量，便于解读 p 值
    n_real, n_gen = len(real), len(gen)
    target = 400
    print(f"Sample sizes -> real: {n_real}, gen/target: {n_gen}/{target}, ratio: {n_gen/target*100:.2f}%")

    # 自适应最小期望频数阈值，提升卡方稳定性
    adaptive_min_expected = max(1, int(0.01 * max(n_gen, 1)))

    # 1) 边际分布指标（包含JS、KL散度等）
    marginals = {col: column_marginal_metrics(real, gen, col, min_expected=adaptive_min_expected) for col in COLUMNS}

    # 2) 两两联合分布指标（包含JS、KL散度）
    joint_metrics = {
        (COLUMNS[0],COLUMNS[1]): pair_joint_metrics(real, gen, COLUMNS[0],COLUMNS[1]),
        (COLUMNS[0],COLUMNS[2]): pair_joint_metrics(real, gen, COLUMNS[0],COLUMNS[2]),
        (COLUMNS[1],COLUMNS[2]): pair_joint_metrics(real, gen, COLUMNS[1],COLUMNS[2]),
    }
    
    # 保持向后兼容：单独提取JS散度
    joint_js = {pair: metrics["JS"] for pair, metrics in joint_metrics.items()}

    # 3) 简单比例对比（替代/补充 p 值展示）
    unique_values = sorted(pd.unique(pd.concat([real[c] for c in COLUMNS] + [gen[c] for c in COLUMNS], ignore_index=True)))
    real_props = proportions(real, COLUMNS, values=unique_values)
    gen_props  = proportions(gen,  COLUMNS, values=unique_values)
    real_props_pct = proportions_to_percent(real_props)
    gen_props_pct  = proportions_to_percent(gen_props)

    # 输出结果
    # print("\n" + "="*80)
    # print("📊 分布差异评估结果")
    # print("="*80)
    
    # print("\n🔍 边际分布指标:")
    # for col, metrics in marginals.items():
    #     print(f"\n列 {col}:")
    #     print(f"  JS散度: {metrics['JS']:.6f}")
    #     # print(f"  KL散度 (real||gen): {metrics['KL_real_gen']:.6f}  # 模式遗漏程度")
    #     # print(f"  KL散度 (gen||real): {metrics['KL_gen_real']:.6f}  # 虚假模式程度") 
    #     print(f"  对称KL散度: {metrics['KL_symmetric']:.6f}  # 综合评估")
    #     # print(f"  卡方统计量: {metrics['chi2']:.4f}, p值: {metrics['p']:.6f}")
    
    # print("\n🔍 两两联合分布指标:")
    # for pair, metrics in joint_metrics.items():
    #     print(f"\n变量对 {pair[0]}-{pair[1]}:")
    #     print(f"  JS散度: {metrics['JS']:.6f}")
    #     # print(f"  KL散度 (real||gen): {metrics['KL_real_gen']:.6f}  # 关联模式遗漏")
    #     # print(f"  KL散度 (gen||real): {metrics['KL_gen_real']:.6f}  # 虚假关联模式")
    #     print(f"  对称KL散度: {metrics['KL_symmetric']:.6f}  # 综合关联评估")
    
    # print("\n📈 分布比例对比:")
    # print("真实数据比例:", real_props_pct)
    # print("生成数据比例:", gen_props_pct)
    
    # 简洁的汇总表
    print("\n📋 指标汇总表:")
    print("-" * 80)
    print(f"{'指标':<20} {COLUMNS[0]}:<12 {COLUMNS[1]}:<12 {COLUMNS[2]}:<12")
    print("-" * 80)
    
    metrics_to_show = ['JS', 'KL_symmetric']
    metric_names = {'JS': 'JS散度', 'KL_symmetric': '对称KL散度'}
    
    for metric in metrics_to_show:
        row = f"{metric_names[metric]:<20}"
        for col in COLUMNS:
            if metric == 'p':
                value = f"{marginals[col][metric]:.4f}"
            else:
                value = f"{marginals[col][metric]:.6f}"
            row += f"{value:<12}"
        print(row)
    
    print("\n📋 联合分布指标汇总:")
    print("-" * 80)
    print(f"{'变量对':<20} {'JS散度':<12} {'对称KL散度':<12}")
    print("-" * 80)
    for pair, metrics in joint_metrics.items():
        pair_name = f"{pair[0]}-{pair[1]}"
        print(f"{pair_name:<20} {metrics['JS']:<12.6f} {metrics['KL_symmetric']:<12.6f}")
    print("=" * 80)

if __name__ == "__main__":
    #os.environ["COLUMNS"] = "["nateduc","natdrug","natcrime"]"
    # calculate_metrics(real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_sft_edc.csv")
    # calculate_metrics(real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_edc.csv")
    # calculate_metrics(real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_edc.csv")
    # calculate_metrics(real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_edc.csv")
    calculate_metrics(COLUMNS=["natenvir", "natheal", "natcity"], real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_ehc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_none_RAG_sft_ehc_v3.csv")
    calculate_metrics(COLUMNS=["nateduc", "natdrug", "natcrime"], real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_none_RAG_sft_edc_v3.csv")
    #calculate_metrics(COLUMNS=["nateduc", "natdrug", "natcrime"], real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_edc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_edc_v2.csv")
    #calculate_metrics(COLUMNS=["natenvir", "natheal", "natcity"], real_path="/home/cyyuan/FAS_PRO/data/gss/GSS2024_sampled_ehc.csv", gen_path="/home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_ehc_v2.csv")