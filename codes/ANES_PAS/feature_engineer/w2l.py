import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics import cohen_kappa_score
import lightgbm as lgb

# 支持度文本到数值的映射
SUPPORT_MAP = {
    'strongly_oppose': 0,
    'oppose': 1,
    'unsupport': 1,
    'slightly_oppose': 2,
    'neutral': 3,
    'slightly_support': 4,
    'support': 5,
    'strongly_support': 6
}

def standardize_support_level(series, scale_type='text'):
    """
    将不同格式的支持度统一转换为0-5的数值
    
    Args:
        series: 支持度列
        scale_type: 'text' (文本), '1-6' (1到6), '0-5' (0到5)
    """
    if scale_type == 'text':
        # 文本映射到0-6，然后标准化到0-5
        numeric = series.map(SUPPORT_MAP)
        # 将0-6映射到0-5
        return (numeric * 5 / 6).round().astype('Int64')
    elif scale_type == '1-6':
        return (series - 1).astype('Int64')
    elif scale_type == '0-5':
        return series.astype('Int64')
    else:
        return series

def wide_to_long(df_wide, year, candidate_map, scale_type='text'):
    """
    将宽表转换为长表
    
    Args:
        df_wide: 某年的宽表数据
        year: int, 年份
        candidate_map: dict，候选人列名到标准ID的映射
        scale_type: str, 支持度的量表类型
    
    Returns:
        df_long: 长表格式的数据
    """
    # 识别受访者特征列（排除support_和year）
    id_cols = ["respondent_id"] + [
        c for c in df_wide.columns 
        if (c.startswith("demo_") or c.startswith("resp_")) 
        and c != "year"
    ]
    
    # 如果respondent_id不存在，自动生成
    if "respondent_id" not in df_wide.columns:
        df_wide["respondent_id"] = range(len(df_wide))
        id_cols = ["respondent_id"] + [c for c in id_cols if c != "respondent_id"]
    
    rows = []
    for col in df_wide.columns:
        if col.startswith("support_"):
            cand_key = col.replace("support_", "")
            cand_id = candidate_map.get(cand_key, cand_key)
            
            tmp = df_wide[id_cols].copy()
            tmp["candidate_id"] = cand_id
            tmp["year"] = year
            tmp["target_raw"] = df_wide[col]
            
            # 标准化支持度
            tmp["target"] = standardize_support_level(df_wide[col], scale_type)
            
            # 添加缺失指示器
            tmp["target_is_missing"] = tmp["target"].isna().astype(int)
            
            rows.append(tmp)
    
    df_long = pd.concat(rows, axis=0, ignore_index=True)
    return df_long