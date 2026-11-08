import json  
import numpy as np  
import pandas as pd
import time
import random
import sys
import os

# 添加路径以导入get_distribution
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from get_distribution import get_distributions_for_attrs  

# =========================  
# 配置区：列名与2024教师分布信息  
# =========================  
# 属性配置：支持灵活的属性名称和描述映射
# 默认配置 - 可通过 update_attr_config() 函数动态修改
ATTR_CONFIG = {
    "natcity": "national spending on solving the problems of big cities",
    "natenvir": "national spending on improving and protecting the environment", 
    "natheal": "national spending on improving and protecting the nation's health"
}

# 预定义的配置模板
PREDEFINED_CONFIGS = {
    "config_1": {
        "natcity": "national spending on solving the problems of big cities",
        "natenvir": "national spending on improving and protecting the environment", 
        "natheal": "national spending on improving and protecting the nation's health"
    },
    "config_2": {
        "natcrime": "national spending on halting the rising crime rate",
        "natdrug": "national spending on dealing with drug addiction", 
        "nateduc": "national spending on education"
    },
    "config_3": {
        "natfare": "national spending on welfare",
        "natsoc": "national spending on social security", 
        "nataid": "national spending on foreign aid"
    }
}

# 从配置中提取属性名称（保持顺序）
ATTR_NAMES = list(ATTR_CONFIG.keys())
ATTR_A, ATTR_B, ATTR_C = ATTR_NAMES  
levels = ['1','2','3']  # '1'=too little, '2'=about right, '3'=too much (views on national spending)

# 生成动态描述文本的辅助函数
def get_attr_description_text():
    """生成属性描述文本，用于prompt中"""
    descriptions = [f"'{attr}' ({desc})" for attr, desc in ATTR_CONFIG.items()]
    return ", ".join(descriptions)

def get_sample_format_text():
    """生成样本格式描述文本"""
    attr_list = "', '".join(ATTR_NAMES)
    return f"['{attr_list}']"

def update_attr_config(new_config):
    """
    更新属性配置
    
    Args:
        new_config: 新的属性配置字典
                   格式: {"attr_name": "description", ...}
    
    Example:
        update_attr_config({
            "natarms": "national spending on military and defense",
            "nateduc": "national spending on education", 
            "natroad": "national spending on highways and roads"
        })
    """
    global ATTR_CONFIG, ATTR_NAMES, ATTR_A, ATTR_B, ATTR_C
    
    if len(new_config) != 3:
        raise ValueError("Must provide exactly 3 attributes")
    
    ATTR_CONFIG = new_config.copy()
    ATTR_NAMES = list(ATTR_CONFIG.keys())
    ATTR_A, ATTR_B, ATTR_C = ATTR_NAMES
    
    print(f"Updated attribute configuration:")
    for attr, desc in ATTR_CONFIG.items():
        print(f"  {attr}: {desc}")
    print(f"Sample format: {get_sample_format_text()}")

def use_predefined_config(config_name):
    """
    使用预定义的配置
    
    Args:
        config_name: 预定义配置名称 ('config_1', 'config_2', 'config_3')
    
    Example:
        use_predefined_config('config_2')  # 切换到犯罪/毒品/教育配置
    """
    if config_name not in PREDEFINED_CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(PREDEFINED_CONFIGS.keys())}")
    
    config = PREDEFINED_CONFIGS[config_name]
    update_attr_config(config)
    print(f"✅ 已切换到预定义配置: {config_name}")  

def initialize_distributions_from_real_data():
    """
    从get_distribution.py获取真实的计算分布，替换硬编码值
    """
    global P_A_2024, P_B_2024, P_C_2024, P_BC_2024, P_AB_2024, P_AC_2024, Q_2024
    
    # 获取当前属性配置对应的真实分布
    current_attrs = list(ATTR_CONFIG.keys())
    Q_joint, Pa_pred, Pb_pred, Pc_pred, predicted_joints = get_distributions_for_attrs(*current_attrs)
    
    # 更新边际分布
    P_A_2024 = Pa_pred.copy()
    P_B_2024 = Pb_pred.copy() 
    P_C_2024 = Pc_pred.copy()
    
    # 更新二阶交叉分布 - 动态键名
    attr_names = current_attrs
    P_BC_2024 = predicted_joints[f'P_{attr_names[1]}_{attr_names[2]}'].copy()  # P(B,C)
    P_AB_2024 = predicted_joints[f'P_{attr_names[0]}_{attr_names[1]}'].copy()  # P(A,B)
    P_AC_2024 = predicted_joints[f'P_{attr_names[0]}_{attr_names[2]}'].copy()  # P(A,C)
    
    # 更新联合分布
    Q_2024 = Q_joint.copy()
    
    print("🔄 已更新为基于真实数据的分布:")
    print(f"P({current_attrs[0]}): {P_A_2024.round(3).to_dict()}")
    print(f"P({current_attrs[1]}): {P_B_2024.round(3).to_dict()}")
    print(f"P({current_attrs[2]}): {P_C_2024.round(3).to_dict()}")

# 归一化工具  
def normalize_series(p: pd.Series) -> pd.Series:  
    s = float(p.sum())  
    return p / s if s > 0 else p  

# def ensure_tables_normalized():  
#     for T in [P_BC_2024, P_AB_2024, P_AC_2024]:  
#         T /= T.values.sum()  
#     for S in [P_A_2024, P_B_2024, P_C_2024]:  
#         S[:] = normalize_series(S)  

# ensure_tables_normalized()  

# =========================  
# 动态随机种子生成工具  
# =========================
def generate_dynamic_seed():
    """生成基于时间戳和随机数的动态种子，确保每次运行都不同"""
    return int(time.time() * 1000) % 1000000 + random.randint(1, 9999)

# =========================  
# 构造联合分布 Q(A,B,C) 的 IPF/熵投影工具  
# =========================  
def ipf_joint_marginals(Pa, Pb, Pc, iters=500, eps=1e-12):  
    # 初始独立  
    Q = np.zeros((3,3,3), dtype=float)  
    for i,a in enumerate(levels):  
        for j,b in enumerate(levels):  
            for k,c in enumerate(levels):  
                Q[i,j,k] = Pa[a]*Pb[b]*Pc[c]  
    Q /= Q.sum()  

    for _ in range(iters):  
        # 匹配 Pa（∑_{B,C}）  
        sum_bc = Q.sum(axis=(1,2)) + eps  
        for i in range(3):  
            Q[i,:,:] *= (Pa[levels[i]] / sum_bc[i])  

        # 匹配 Pb（∑_{A,C}）  
        sum_ac = Q.sum(axis=(0,2)) + eps  
        for j in range(3):  
            Q[:,j,:] *= (Pb[levels[j]] / sum_ac[j])  

        # 匹配 Pc（∑_{A,B}）  
        sum_ab = Q.sum(axis=(0,1)) + eps  
        for k in range(3):  
            Q[:,:,k] *= (Pc[levels[k]] / sum_ab[k])  

        Q /= Q.sum()  

    # 平滑  
    alpha = 1e-9  
    Q = (Q + alpha) / (Q + alpha).sum()  
    return Q  

def ipf_with_tables(Pa, Pb, Pc, Pbc=None, Pab=None, Pac=None, iters=800, eps=1e-12):  
    # 初始独立  
    Q = np.zeros((3,3,3), dtype=float)  
    for i,a in enumerate(levels):  
        for j,b in enumerate(levels):  
            for k,c in enumerate(levels):  
                Q[i,j,k] = Pa[a]*Pb[b]*Pc[c]  
    Q /= Q.sum()  

    for _ in range(iters):  
        # 依次匹配二阶交叉  
        if Pbc is not None:  
            target = Pbc.values  
            proj = Q.sum(axis=0) + eps    # sum_A Q -> (B,C)  
            Q *= (target / proj)[None, :, :]  
        if Pab is not None:  
            target = Pab.values  
            proj = Q.sum(axis=2) + eps    # sum_C Q -> (A,B)  
            Q *= (target / proj)[:, :, None]  
        if Pac is not None:  
            target = Pac.values  
            proj = Q.sum(axis=1) + eps    # sum_B Q -> (A,C)  
            Q *= (target / proj)[:, None, :]  

        # 匹配三条边际  
        sum_bc = Q.sum(axis=(1,2)) + eps  
        for i in range(3):  
            Q[i,:,:] *= (Pa[levels[i]] / sum_bc[i])  

        sum_ac = Q.sum(axis=(0,2)) + eps  
        for j in range(3):  
            Q[:,j,:] *= (Pb[levels[j]] / sum_ac[j])  

        sum_ab = Q.sum(axis=(0,1)) + eps  
        for k in range(3):  
            Q[:,:,k] *= (Pc[levels[k]] / sum_ab[k])  

        Q = np.clip(Q, 0.0, None)  
        Q /= Q.sum()  

    alpha = 1e-9  
    Q = (Q + alpha) / (Q + alpha).sum()  
    return Q  

def sample_from_joint(Q, n=20, seed=None):  
    if seed is None:
        seed = generate_dynamic_seed()
    rng = np.random.default_rng(seed)  
    probs = (Q / Q.sum()).flatten()  
    idx = rng.choice(len(probs), size=n, replace=True, p=probs)  
    samples = []  
    for m in idx:  
        i = m // 9  
        j = (m % 9) // 3  
        k = m % 3  
        samples.append([levels[i], levels[j], levels[k]])  
    return samples  

def samples_to_multiline(samples):  
    return "\n".join([str(row).replace(" ", "") for row in samples])  

# =========================  
# 子群体指标：点估计与95%CI（Wilson）  
# =========================  
def wilson_interval(x, n, z=1.96):  
    if n == 0:  
        return (0.0, 0.0, 0.0)  
    phat = x / n  
    denom = 1 + z**2 / n  
    center = (phat + z**2/(2*n)) / denom  
    half = z * np.sqrt((phat*(1-phat) + z**2/(4*n)) / n) / denom  
    return phat, max(0.0, center - half), min(1.0, center + half)  

def estimate_conditional(samples, attr_given_name, given_value, attr_target_name, target_value):  
    # 动态列名 -> 索引  
    col_idx = {ATTR_A:0, ATTR_B:1, ATTR_C:2}  
    j = col_idx[attr_given_name]  
    k = col_idx[attr_target_name]  
    universe = [s for s in samples if s[j] == given_value]  
    x = sum(1 for s in universe if s[k] == target_value)  
    n = len(universe)  
    phat, lo, hi = wilson_interval(x, n)  
    return phat, lo, hi, x, n  

# =========================  
# 生成五类微调数据（动态列名）  
# =========================  
def make_task_batch_generation(Q, n=20, seed=None):  
    if seed is None:
        seed = generate_dynamic_seed()
    samples = sample_from_joint(Q, n=n, seed=seed)  
    sample_format = get_sample_format_text()
    attr_descriptions = get_attr_description_text()
    item = {  
        "instruction": f"Generate a batch of {n} respondent samples. Each sample format is {sample_format}, with values '1', '2', or '3' representing respondents' views on national spending: '1'=too little, '2'=about right, '3'=too much. The attributes represent: {attr_descriptions}. Only output {n} rows in the format ['1','2','3'], do not add explanations.", 
        "input": "",  
        "output": samples_to_multiline(samples)  
    }  
    return item, samples  

def make_task_subgroup_qa(samples, given_attr=ATTR_B, given_val='3', target_attr=ATTR_C, target_val='3'):  
    phat, lo, hi, x, n = estimate_conditional(samples, given_attr, given_val, target_attr, target_val)  
    sample_format = get_sample_format_text()
    given_desc = ATTR_CONFIG[given_attr]
    target_desc = ATTR_CONFIG[target_attr]
    item = {  
        "instruction": f"Calculate the point estimate and 95% confidence interval for P({target_attr}='{target_val}' | {given_attr}='{given_val}') based on the following sample (each row {sample_format}). Here {given_attr} represents {given_desc} and {target_attr} represents {target_desc}. Values: '1'=too little, '2'=about right, '3'=too much. Output only: point, ci_low, ci_high.",
        "input": samples_to_multiline(samples),  
        "output": f"{phat:.2f}, {lo:.2f}, {hi:.2f}"  
    }  
    return item  

def make_task_prior_sampling(priors, n=20, seed=None):  
    if seed is None:
        seed = generate_dynamic_seed()
    # priors 的键为列名，内部为 level->prob  
    Pa = pd.Series(priors[ATTR_A])  
    Pb = pd.Series(priors[ATTR_B])  
    Pc = pd.Series(priors[ATTR_C])  
    Q = ipf_joint_marginals(Pa, Pb, Pc, iters=300)  
    samples = sample_from_joint(Q, n=n, seed=seed)  
    sample_format = get_sample_format_text()
    attr_descriptions = get_attr_description_text()
    # 动态生成先验描述
    prior_texts = []
    for i, attr in enumerate(ATTR_NAMES):
        prob = [Pa, Pb, Pc][i]['3']
        prior_texts.append(f"P({attr}='3')≈{prob:.2f}")
    prior_description = "; ".join(prior_texts)
    
    item = {  
        "instruction": f"Generate {n} samples, each in format {sample_format} with values in {{'1','2','3'}} representing views on national spending ('1'=too little, '2'=about right, '3'=too much). The attributes represent: {attr_descriptions}. Distribution priors (overall proportions, ±0.05 tolerance): {prior_description}. Try to match the prior distributions and only output {n} rows in ['1','2','3'] format.", 
        "input": "",  
        "output": samples_to_multiline(samples)  
    }  
    return item, samples  

def make_task_iterative_fix(current_stats, k=6, seed=None, target_attr=ATTR_B, target_level='3', boost_prob=0.7):  
    # 目标：提升 target_attr==target_level 的比例，生成 k 条替换样本  
    if seed is None:
        seed = generate_dynamic_seed()
    rng = np.random.default_rng(seed)  
    # 构造偏置概率：target_attr 的 level='3' 占 boost_prob，其余均分剩余  
    if target_attr == ATTR_A:  
        a_probs = np.array([(1-boost_prob)/2, (1-boost_prob)/2, boost_prob])  
        b_probs = P_B_2024.values  
        c_probs = P_C_2024.values  
    elif target_attr == ATTR_B:  
        b_probs = np.array([(1-boost_prob)/2, (1-boost_prob)/2, boost_prob])  
        a_probs = P_A_2024.values  
        c_probs = P_C_2024.values  
    else:  
        c_probs = np.array([(1-boost_prob)/2, (1-boost_prob)/2, boost_prob])  
        a_probs = P_A_2024.values  
        b_probs = P_B_2024.values  

    out = []  
    for _ in range(k):  
        a = levels[rng.choice(3, p=a_probs)]  
        b = levels[rng.choice(3, p=b_probs)]  
        c = levels[rng.choice(3, p=c_probs)]  
        out.append([a,b,c])  

    sample_format = get_sample_format_text()
    attr_descriptions = get_attr_description_text()
    target_desc = ATTR_CONFIG[target_attr]
    item = {  
        "instruction": f"Based on statistics from the previous sample batch, current P({target_attr}='{target_level}') is below the target range. Please generate {k} new samples (format: {sample_format}) to replace old samples and improve the proportion of P({target_attr}='{target_level}'). Here {target_attr} represents {target_desc}. The attributes represent: {attr_descriptions}. Each attribute value is '1', '2', or '3' representing views on national spending ('1'=too little, '2'=about right, '3'=too much). Only output {k} rows, each in format ['1','2','3'].",
        "input": json.dumps(current_stats, ensure_ascii=False),  
        "output": samples_to_multiline(out)  
    }  
    return item  

def make_task_year_scenario(priors_center, n=20, seed=None):  
    # priors_center: 三个列的“目标中心值”给定为各自 '3' 的概率；其余 '1'/'2' 按参考比例切分  
    def split_center(p3_center, base12):  
        p3 = float(p3_center)  
        p12 = 1 - p3  
        return (p12*base12[0], p12*base12[1], p3)  

    # 参考比例用 2024 教师边际（也可改为最近年）  
    baseA = [P_A_2024['1']/(P_A_2024['1']+P_A_2024['2']),  
             P_A_2024['2']/(P_A_2024['1']+P_A_2024['2'])]  
    baseB = [P_B_2024['1']/(P_B_2024['1']+P_B_2024['2']),  
             P_B_2024['2']/(P_B_2024['1']+P_B_2024['2'])]  
    baseC = [P_C_2024['1']/(P_C_2024['1']+P_C_2024['2']),  
             P_C_2024['2']/(P_C_2024['1']+P_C_2024['2'])]  

    a1,a2,a3 = split_center(priors_center[ATTR_A], baseA)  
    b1,b2,b3 = split_center(priors_center[ATTR_B], baseB)  
    c1,c2,c3 = split_center(priors_center[ATTR_C], baseC)  

    Pa = pd.Series({'1': a1, '2': a2, '3': a3})  
    Pb = pd.Series({'1': b1, '2': b2, '3': b3})  
    Pc = pd.Series({'1': c1, '2': c2, '3': c3})  

    if seed is None:
        seed = generate_dynamic_seed()
    Q = ipf_joint_marginals(Pa, Pb, Pc, iters=300)  
    samples = sample_from_joint(Q, n=n, seed=seed)  
    sample_format = get_sample_format_text()
    attr_descriptions = get_attr_description_text()
    # 动态生成先验中心值描述
    center_texts = []
    for i, attr in enumerate(ATTR_NAMES):
        prob = [Pa, Pb, Pc][i]['3']
        center_texts.append(f"P({attr}='3')≈{prob:.2f}")
    center_description = "; ".join(center_texts)
    
    item = {  
        "instruction": f"Scenario: Policy changes in the target year. Generate {n} samples, each in format {sample_format} with values '1'/'2'/'3' representing views on national spending ('1'=too little, '2'=about right, '3'=too much). The attributes represent: {attr_descriptions}. Prior center values: {center_description}. Only output {n} rows.",  
        "input": "",  
        "output": samples_to_multiline(samples)  
    }  
    return item  

# =========================  
# 构造 2024 教师联合分布 Q（多张二阶交叉 + 三边际）  
# =========================  
# Q_2024 = ipf_with_tables(P_A_2024, P_B_2024, P_C_2024,  
#                          Pbc=P_BC_2024, Pab=P_AB_2024, Pac=P_AC_2024,  
#                          iters=800)  

# =========================
# 按比例批量生成微调数据集
# =========================
def generate_realistic_priors(use_real_base=True):
    """
    生成更真实的先验参数，基于当前真实分布进行合理变化
    
    Args:
        use_real_base: 是否基于真实分布作为基准
    
    Returns:
        dict: 先验参数字典
    """
    if use_real_base:
        # 基于真实2024分布进行小幅变化
        base_a = P_A_2024.copy()
        base_b = P_B_2024.copy() 
        base_c = P_C_2024.copy()
        
        # 添加小幅随机变化（±10%以内）
        variation = 0.10
        priors = {}
        for attr, base_dist in [(ATTR_A, base_a), (ATTR_B, base_b), (ATTR_C, base_c)]:
            attr_priors = {}
            for level in levels:
                base_val = float(base_dist[level])
                # 在基准值周围变化
                delta = random.uniform(-variation * base_val, variation * base_val)
                new_val = max(0.01, base_val + delta)  # 确保不为负
                attr_priors[level] = new_val
            priors[attr] = attr_priors
        
        # 归一化
        for attr in [ATTR_A, ATTR_B, ATTR_C]:
            total = sum(priors[attr].values())
            for level in levels:
                priors[attr][level] /= total
    else:
        # 原来的随机生成方式
        priors = {
            ATTR_A: {'1': random.uniform(0.3, 0.7), '2': random.uniform(0.1, 0.4), '3': random.uniform(0.1, 0.4)},
            ATTR_B: {'1': random.uniform(0.4, 0.8), '2': random.uniform(0.1, 0.3), '3': random.uniform(0.05, 0.25)},
            ATTR_C: {'1': random.uniform(0.5, 0.8), '2': random.uniform(0.1, 0.3), '3': random.uniform(0.02, 0.15)},
        }
        # 归一化
        for attr in [ATTR_A, ATTR_B, ATTR_C]:
            total = sum(priors[attr].values())
            for level in levels:
                priors[attr][level] /= total
    
    return priors

def generate_realistic_scenario_priors():
    """
    生成更真实的情景先验，基于历史趋势和政策假设
    """
    # 基于真实'3'比例进行政策情景变化
    real_ratios = {
        ATTR_A: float(P_A_2024['3']),  # 当前"过多"比例
        ATTR_B: float(P_B_2024['3']), 
        ATTR_C: float(P_C_2024['3'])
    }
    
    # 政策情景：可能增加或减少对某领域的投入
    scenario_priors = {}
    for attr, real_ratio in real_ratios.items():
        # 政策变化：-50%到+100%的变化范围
        change_factor = random.uniform(0.5, 2.0)
        new_ratio = min(0.4, max(0.01, real_ratio * change_factor))  # 限制在合理范围
        scenario_priors[attr] = new_ratio
    
    return scenario_priors

def generate_balanced_finetune_dataset(total_samples=1000, output_file=None, custom_ratios=None, use_realistic_priors=True):
    """
    按推荐比例生成平衡的微调数据集
    
    Args:
        total_samples: 总样本数量
        output_file: 输出文件路径（可选）
        custom_ratios: 自定义比例字典（可选）
                      格式: {'batch_generation': 0.5, 'prior_sampling': 0.25, ...}
        use_realistic_priors: 是否使用基于真实数据的先验参数
    
    Returns:
        list: 包含所有训练样本的列表
    """
    # 🔥 关键修复：使用真实计算的分布数据替换硬编码值
    print("🔄 正在初始化真实分布数据...")
    initialize_distributions_from_real_data()
    # 默认推荐比例
    default_ratios = {
        'batch_generation': 0.50,    # 核心任务：规范化输出 + 符合分布
        'prior_sampling': 0.25,      # 分布理解：先验采样
        'year_scenario': 0.15,       # 分布适应：年份情景
        'subgroup_qa': 0.07,         # 统计理解：子群体分析  
        'iterative_fix': 0.03        # 辅助任务：迭代修正
    }
    
    # 使用自定义比例或默认比例
    ratios = custom_ratios if custom_ratios else default_ratios
    
    # 计算各类型样本数量
    counts = {}
    remaining = total_samples
    for task_type, ratio in ratios.items():
        if task_type == list(ratios.keys())[-1]:  # 最后一个任务取剩余数量
            counts[task_type] = remaining
        else:
            counts[task_type] = int(total_samples * ratio)
            remaining -= counts[task_type]
    
    print(f"Generating {total_samples} samples with distribution:")
    for task_type, count in counts.items():
        print(f"  {task_type}: {count} samples ({count/total_samples*100:.1f}%)")
    
    dataset = []
    
    # 1. 批量样本生成 (核心任务)
    print("Generating batch generation samples...")
    for i in range(counts['batch_generation']):
        item, samples = make_task_batch_generation(Q_2024, n=20)
        dataset.append(item)
        
        # 每10个批量生成样本配1个对应的子群体问答
        if i < counts['subgroup_qa'] and i % (counts['batch_generation'] // max(1, counts['subgroup_qa'])) == 0:
            # 随机选择分析的属性和值
            given_attrs = [ATTR_A, ATTR_B, ATTR_C]
            target_attrs = [ATTR_A, ATTR_B, ATTR_C]
            given_attr = random.choice(given_attrs)
            target_attr = random.choice([attr for attr in target_attrs if attr != given_attr])
            
            item_qa = make_task_subgroup_qa(samples, 
                                          given_attr=given_attr, given_val='3', 
                                          target_attr=target_attr, target_val='3')
            dataset.append(item_qa)
    
    # 2. 先验分布采样
    print("Generating prior sampling samples...")
    for i in range(counts['prior_sampling']):
        # 70%使用基于真实数据的先验，30%使用随机先验
        use_real_base = (i < counts['prior_sampling'] * 0.7) if use_realistic_priors else False
        priors = generate_realistic_priors(use_real_base=use_real_base)
        
        item, _ = make_task_prior_sampling(priors, n=20)
        dataset.append(item)
    
    # 3. 年份情景先验
    print("Generating year scenario samples...")
    for i in range(counts['year_scenario']):
        if use_realistic_priors and i < counts['year_scenario'] * 0.5:
            # 50%使用基于真实数据的情景变化
            priors_center = generate_realistic_scenario_priors()
        else:
            # 50%使用随机情景
            priors_center = {
                ATTR_A: random.uniform(0.10, 0.30),  # 城市投入过多的比例
                ATTR_B: random.uniform(0.05, 0.20),  # 环境投入过多的比例
                ATTR_C: random.uniform(0.03, 0.12),  # 健康投入过多的比例
            }
        item = make_task_year_scenario(priors_center, n=20)
        dataset.append(item)
    
    # 4. 迭代修正
    print("Generating iterative fix samples...")
    for i in range(counts['iterative_fix']):
        # 随机选择目标属性和当前比例
        target_attrs = [ATTR_A, ATTR_B, ATTR_C]
        target_attr = random.choice(target_attrs)
        current_ratio = random.uniform(0.05, 0.25)
        target_range = [current_ratio + 0.05, current_ratio + 0.15]
        
        current_stats = {
            "current": {target_attr: {"'3'": current_ratio}},
            "target_range": {target_attr: target_range}
        }
        item = make_task_iterative_fix(current_stats, k=6, target_attr=target_attr, target_level='3')
        dataset.append(item)
    
    # 打乱数据集顺序
    random.shuffle(dataset)
    
    # 保存到文件
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"\nDataset saved to {output_file}")
    
    print(f"\nGenerated {len(dataset)} total training samples")
    return dataset

# =========================
# 批量生成微调数据集
# =========================
def generate_finetune_dataset(num_samples_per_type=50, output_file=None):
    """
    批量生成五类微调数据，每次调用都会生成不同的样本
    
    Args:
        num_samples_per_type: 每种类型生成的样本数量
        output_file: 输出文件路径（可选）
    """
    # 🔥 关键修复：使用真实计算的分布数据替换硬编码值
    print("🔄 正在初始化真实分布数据...")
    initialize_distributions_from_real_data()
    dataset = []
    
    # a) 批量样本生成
    for i in range(num_samples_per_type):
        item_a, samples_a = make_task_batch_generation(Q_2024, n=20)
        dataset.append(item_a)
        
        # b) 基于样本a生成子群体问答
        item_b = make_task_subgroup_qa(samples_a, given_attr=ATTR_B, given_val='3', target_attr=ATTR_C, target_val='3')
        dataset.append(item_b)
    
    # c) 先验分布采样
    for i in range(num_samples_per_type):
        # 随机生成不同的先验参数
        priors_c = {
            ATTR_A: {'1': random.uniform(0.2, 0.4), '2': random.uniform(0.2, 0.4), '3': random.uniform(0.2, 0.6)},
            ATTR_B: {'1': random.uniform(0.3, 0.5), '2': random.uniform(0.2, 0.4), '3': random.uniform(0.1, 0.5)},
            ATTR_C: {'1': random.uniform(0.2, 0.4), '2': random.uniform(0.2, 0.4), '3': random.uniform(0.2, 0.6)},
        }
        # 归一化
        for attr in [ATTR_A, ATTR_B, ATTR_C]:
            total = sum(priors_c[attr].values())
            for level in ['1', '2', '3']:
                priors_c[attr][level] /= total
        
        item_c, _ = make_task_prior_sampling(priors_c, n=20)
        dataset.append(item_c)
    
    # d) 迭代修正
    for i in range(num_samples_per_type):
        # 随机生成不同的统计信息
        target_attrs = [ATTR_A, ATTR_B, ATTR_C]
        target_attr = random.choice(target_attrs)
        current_ratio = random.uniform(0.1, 0.3)
        target_range = [current_ratio + 0.1, current_ratio + 0.25]
        
        current_stats_d = {
            "current": {target_attr: {"'3'": current_ratio}},
            "target_range": {target_attr: target_range}
        }
        item_d = make_task_iterative_fix(current_stats_d, k=6, target_attr=target_attr, target_level='3', boost_prob=0.70)
        dataset.append(item_d)
    
    # e) 年份情景先验
    for i in range(num_samples_per_type):
        priors_center_e = {
            ATTR_A: random.uniform(0.25, 0.45),
            ATTR_B: random.uniform(0.20, 0.40), 
            ATTR_C: random.uniform(0.25, 0.45),
        }
        item_e = make_task_year_scenario(priors_center_e, n=20)
        dataset.append(item_e)
    
    # 打乱数据集顺序
    random.shuffle(dataset)
    
    # 保存到文件（可选）
    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        print(f"Dataset saved to {output_file}")
    
    return dataset

# =========================
# 示例：生成并打印 5 类数据（使用你的列名）
# =========================

def main():
    # 🔥 关键修复：使用真实计算的分布数据替换硬编码值
    print("🔄 正在初始化真实分布数据...")
    initialize_distributions_from_real_data()
    
    print("=== Current Configuration ===")
    print("Default attributes:")
    for attr, desc in ATTR_CONFIG.items():
        print(f"  {attr}: {desc}")
    print(f"Sample format: {get_sample_format_text()}")
    
    print("\n=== Single Sample Examples ===")
    # a) 批量样本生成（20条）- 使用动态seed
    item_a, samples_a = make_task_batch_generation(Q_2024, n=20)
    print(json.dumps(item_a, ensure_ascii=False, indent=2))

    # b) 子群体指标问答（基于上一批样本）
    item_b = make_task_subgroup_qa(samples_a, given_attr=ATTR_B, given_val='3', target_attr=ATTR_C, target_val='3')
    print("\n" + json.dumps(item_b, ensure_ascii=False, indent=2))

    # c) 先验分布采样（按边际先验）
    priors_c = {
        ATTR_A: {'1': 0.30, '2': 0.30, '3': 0.40},  # P(A='3')≈0.40
        ATTR_B: {'1': 0.40, '2': 0.30, '3': 0.30},  # P(B='3')≈0.30
        ATTR_C: {'1': 0.30, '2': 0.35, '3': 0.35},  # P(C='3')≈0.35
    }
    item_c, samples_c = make_task_prior_sampling(priors_c, n=20)
    print("\n" + json.dumps(item_c, ensure_ascii=False, indent=2))

    # d) 迭代修正（给出6条替换样本以提升某列的 '3'）
    current_stats_d = {
        "current": {ATTR_B: {"'3'": 0.20}},
        "target_range": {ATTR_B: [0.25, 0.35]}
    }
    item_d = make_task_iterative_fix(current_stats_d, k=6, target_attr=ATTR_B, target_level='3', boost_prob=0.70)
    print("\n" + json.dumps(item_d, ensure_ascii=False, indent=2))

    # e) 年份情景先验（给中心概率，1/2 由参考比例切分）
    priors_center_e = {
        ATTR_B: 0.35,  # e.g., 环境=3 的中心概率 ~0.35
        ATTR_A: 0.38,  # e.g., 城市=3 的中心概率 ~0.38
        ATTR_C: 0.36,  # e.g., 健康=3 的中心概率 ~0.36
    }
    item_e = make_task_year_scenario(priors_center_e, n=20)
    print("\n" + json.dumps(item_e, ensure_ascii=False, indent=2))
    
    print("\n\n=== Balanced Dataset Generation Example ===")
    # 生成按推荐比例的训练数据集
    balanced_dataset = generate_balanced_finetune_dataset(
        total_samples=500, 
        output_file="balanced_finetune_dataset.json"
    )
if __name__ == "__main__":
    main()




    # print("\n\n=== Different Attributes Example ===")
    # # 演示使用不同的属性配置
    # print("Changing to different attributes...")
    # update_attr_config({
    #     "natarms": "national spending on military and defense",
    #     "nateduc": "national spending on education", 
    #     "natroad": "national spending on highways and roads"
    # })
    
    # # 使用新配置生成一个示例
    # item_different, _ = make_task_batch_generation(Q_2024, n=5)
    # print("\nExample with new attributes:")
    # print(json.dumps(item_different, ensure_ascii=False, indent=2))