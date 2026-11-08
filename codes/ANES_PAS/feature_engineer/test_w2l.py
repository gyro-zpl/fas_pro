"""
宽表转长表完整测试样例
演示如何使用w2l模块处理ANES数据
"""

import pandas as pd
import numpy as np
from w2l import wide_to_long, build_interactions, prepare_candidate_metadata
from w2l import standardize_support_level
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import log_loss, accuracy_score, cohen_kappa_score
import lightgbm as lgb

# ========== 步骤1: 创建模拟的宽表数据 ==========
print("=" * 60)
print("步骤1: 创建模拟数据")
print("=" * 60)

# 2012年数据
df_2012 = pd.DataFrame({
    'resp_age': [10, 25, 35, 45, 55, 22, 33, 44, 28, 38],
    'resp_race': ['white', 'black', 'white', 'hispanic', 'white', 'black', 'white', 'asian', 'white', 'black'],
    'resp_party': ['Republican', 'Democrat', 'Republican', 'Democrat', 'Republican', 
                   'Democrat', 'Republican', 'Democrat', 'Independent', 'Democrat'],
    'support_trump': ['support', 'unsupport', 'support', 'neutral', 'support', 
                      'unsupport', 'support', 'unsupport', 'neutral', 'unsupport'],
    'support_obama': ['unsupport', 'support', 'neutral', 'support', 'unsupport', 
                      'support', 'unsupport', 'support', 'neutral', 'support'],
    'support_clinton': ['neutral', 'support', 'unsupport', 'support', 'neutral', 
                        'support', 'unsupport', 'support', 'support', 'support']
})

# 2014年数据
df_2014 = pd.DataFrame({
    'resp_age': [23, 34, 44, 54, 29, 39, 49, 31, 41, 51],
    'resp_race': ['black', 'white', 'hispanic', 'white', 'black', 'asian', 'white', 'black', 'white', 'hispanic'],
    'resp_party': ['Democrat', 'Republican', 'Democrat', 'Republican', 'Democrat', 
                   'Republican', 'Democrat', 'Independent', 'Republican', 'Democrat'],
    'support_rubio': ['unsupport', 'support', 'unsupport', 'support', 'neutral', 
                      'support', 'unsupport', 'neutral', 'support', 'unsupport'],
    'support_harris': ['neutral', 'unsupport', 'support', 'unsupport', 'support', 
                       'neutral', 'support', 'support', 'unsupport', 'support'],
    'support_jackson': ['unsupport', 'neutral', 'unsupport', 'neutral', 'unsupport', 
                        'unsupport', 'neutral', 'unsupport', 'neutral', 'unsupport']
})

# 2016年数据（用于测试）
df_2016 = pd.DataFrame({
    'resp_age': [50, 30, 40, 60, 35, 45, 55, 32, 42, 52],
    'resp_race': ['yellow', 'white', 'black', 'white', 'hispanic', 'white', 'black', 'asian', 'white', 'black'],
    'resp_party': ['Republican', 'Democrat', 'Republican', 'Republican', 'Democrat', 
                   'Republican', 'Democrat', 'Independent', 'Republican', 'Democrat'],
    'support_biden': ['unsupport', 'support', 'neutral', 'unsupport', 'support', 
                      'neutral', 'support', 'support', 'unsupport', 'support'],
    'support_trump': ['support', 'unsupport', 'support', 'support', 'unsupport', 
                      'support', 'unsupport', 'neutral', 'support', 'unsupport'],
    'support_jason': ['neutral', 'neutral', 'unsupport', 'neutral', 'unsupport', 
                      'unsupport', 'neutral', 'unsupport', 'neutral', 'neutral']
})

print(f"2012年数据: {len(df_2012)}个受访者, {df_2012.filter(like='support').shape[1]}个候选人")
print(f"2014年数据: {len(df_2014)}个受访者, {df_2014.filter(like='support').shape[1]}个候选人")
print(f"2016年数据: {len(df_2016)}个受访者, {df_2016.filter(like='support').shape[1]}个候选人")
print()

# ========== 步骤2: 转换为长表 ==========
print("=" * 60)
print("步骤2: 宽表转长表")
print("=" * 60)

# 定义候选人映射（将列名映射到标准化的候选人ID）
candidate_map_2012 = {"trump": "Trump", "obama": "Obama", "clinton": "Clinton"}
candidate_map_2014 = {"rubio": "Rubio", "harris": "Harris", "jackson": "Jackson"}
candidate_map_2016 = {"biden": "Biden", "trump": "Trump", "jason": "Jason"}

# 转换为长表
long_2012 = wide_to_long(df_2012, year=2012, candidate_map=candidate_map_2012, scale_type='text')
long_2014 = wide_to_long(df_2014, year=2014, candidate_map=candidate_map_2014, scale_type='text')
long_2016 = wide_to_long(df_2016, year=2016, candidate_map=candidate_map_2016, scale_type='text')

# 合并所有年份
all_long = pd.concat([long_2012, long_2014, long_2016], ignore_index=True)

print(f"长表总行数: {len(all_long)} (受访者 × 候选人)")
print(f"长表列名: {list(all_long.columns)}")
print("\n长表样例（前5行）:")
print(all_long.head())
print()

# ========== 步骤3: 添加候选人元数据 ==========
print("=" * 60)
print("步骤3: 添加候选人元数据")
print("=" * 60)

# 创建候选人元数据
candidate_metadata = [
    {"candidate_id": "Trump", "cand_party": "Republican", "cand_age": 66, "cand_race": "white", "cand_ideology_score": 7},
    {"candidate_id": "Obama", "cand_party": "Democrat", "cand_age": 51, "cand_race": "black", "cand_ideology_score": 3},
    {"candidate_id": "Clinton", "cand_party": "Democrat", "cand_age": 65, "cand_race": "white", "cand_ideology_score": 3},
    {"candidate_id": "Rubio", "cand_party": "Republican", "cand_age": 43, "cand_race": "hispanic", "cand_ideology_score": 7},
    {"candidate_id": "Harris", "cand_party": "Democrat", "cand_age": 50, "cand_race": "black", "cand_ideology_score": 2},
    {"candidate_id": "Jackson", "cand_party": "Democrat", "cand_age": 73, "cand_race": "black", "cand_ideology_score": 2},
    {"candidate_id": "Biden", "cand_party": "Democrat", "cand_age": 74, "cand_race": "white", "cand_ideology_score": 4},
    {"candidate_id": "Jason", "cand_party": "Republican", "cand_age": 45, "cand_race": "white", "cand_ideology_score": 6}
]

candidate_meta = prepare_candidate_metadata(candidate_metadata)
print("候选人元数据:")
print(candidate_meta)
print()

# 合并候选人元数据
all_long = all_long.merge(candidate_meta, on="candidate_id", how="left")

# ========== 步骤4: 构造交互特征 ==========
print("=" * 60)
print("步骤4: 构造交互特征")
print("=" * 60)

# 添加受访者意识形态（示例：基于党派的简单映射）
ideology_map = {"Republican": 7, "Democrat": 3, "Independent": 5}
all_long["resp_ideology"] = all_long["resp_party"].map(ideology_map)

# 构造交互特征
all_long = build_interactions(all_long)

print(f"添加交互特征后的列: {list(all_long.columns)}")
print("\n包含交互特征的样例:")
print(all_long[['respondent_id', 'candidate_id', 'year', 'party_alignment', 
                'ideology_distance', 'age_gap', 'race_alignment', 'target']].head(10))
print()

# ========== 步骤5: 准备训练和测试数据 ==========
print("=" * 60)
print("步骤5: 准备训练和测试数据")
print("=" * 60)

# 训练集: 2012-2014年
train_df = all_long[all_long.year < 2016].copy()
# 测试集: 2016年
test_df = all_long[all_long.year == 2016].copy()

# 删除缺失目标值的行
train_df = train_df[train_df['target'].notna()].copy()
test_df = test_df[test_df['target'].notna()].copy()

print(f"训练集大小: {len(train_df)} (年份: {train_df.year.unique()})")
print(f"测试集大小: {len(test_df)} (年份: {test_df.year.unique()})")
print(f"训练集候选人: {sorted(train_df.candidate_id.unique())}")
print(f"测试集候选人: {sorted(test_df.candidate_id.unique())}")
print()

# ========== 步骤6: 特征工程 ==========
print("=" * 60)
print("步骤6: 特征工程")
print("=" * 60)

# 排除的列
exclude_cols = ["respondent_id", "candidate_id", "year", "target", "target_raw", "target_is_missing"]
feature_cols = [c for c in train_df.columns if c not in exclude_cols]

print(f"特征列: {feature_cols}")
print()

# 区分数值和类别特征
num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
cat_cols = [c for c in feature_cols if c not in num_cols]

print(f"数值特征({len(num_cols)}): {num_cols}")
print(f"类别特征({len(cat_cols)}): {cat_cols}")
print()

# 处理缺失值
for c in num_cols:
    med = train_df[c].median()
    train_df[c] = train_df[c].fillna(med)
    test_df[c] = test_df[c].fillna(med)

for c in cat_cols:
    train_df[c] = train_df[c].fillna("MISSING").astype(str)
    test_df[c] = test_df[c].fillna("MISSING").astype(str)

# 编码类别特征
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_df[cat_cols] = enc.fit_transform(train_df[cat_cols])
test_df[cat_cols] = enc.transform(test_df[cat_cols])

# 准备训练和测试数据
X_train = train_df[feature_cols]
y_train = train_df["target"].astype(int)
X_test = test_df[feature_cols]
y_test = test_df["target"].astype(int)

print(f"X_train shape: {X_train.shape}")
print(f"y_train分布: {np.bincount(y_train)}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test分布: {np.bincount(y_test)}")
print()

# ========== 步骤7: 模型训练 ==========
print("=" * 60)
print("步骤7: LightGBM模型训练")
print("=" * 60)

# 确定类别数
num_classes = int(y_train.max()) + 1
print(f"分类数: {num_classes}")

# LightGBM参数
params = {
    "objective": "multiclass",
    "num_class": num_classes,
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "seed": 42,
    "verbosity": -1
}

# 使用Leave-One-Group-Out交叉验证（按年份分组）
print("\n使用Leave-One-Group-Out交叉验证（按年份）...")
logo = LeaveOneGroupOut()
groups = train_df["year"].values

oof_probs = np.zeros((len(train_df), num_classes))
best_iters = []

for fold_idx, (tr_idx, val_idx) in enumerate(logo.split(X_train, y_train, groups)):
    X_tr, y_tr = X_train.iloc[tr_idx], y_train.iloc[tr_idx]
    X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
    
    val_year = train_df.iloc[val_idx]['year'].unique()[0]
    
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)
    dval = lgb.Dataset(X_val, label=y_val, reference=dtr, categorical_feature=cat_cols, free_raw_data=False)
    
    model = lgb.train(
        params, 
        dtr, 
        num_boost_round=500,
        valid_sets=[dtr, dval],
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(100)]
    )
    
    best_iters.append(model.best_iteration or 100)
    oof_probs[val_idx, :] = model.predict(X_val, num_iteration=model.best_iteration)
    
    preds = oof_probs[val_idx, :].argmax(axis=1)
    # 指定所有可能的标签以避免类别数不匹配
    ll = log_loss(y_val, oof_probs[val_idx, :], labels=list(range(num_classes)))
    acc = accuracy_score(y_val, preds)
    qwk = cohen_kappa_score(y_val, preds, weights="quadratic")
    
    print(f"[Fold {fold_idx} - Year {val_year}] logloss={ll:.4f} | acc={acc:.4f} | qwk={qwk:.4f} | best_iter={model.best_iteration}")

print()

# 在全训练集上训练最终模型
print("训练最终模型...")
final_iter = int(np.median(best_iters)) if best_iters else 100
print(f"使用中位数迭代数: {final_iter}")

dfull = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
final_model = lgb.train(params, dfull, num_boost_round=final_iter)

# ========== 步骤8: 测试集评估 ==========
print("=" * 60)
print("步骤8: 测试集评估（2016年）")
print("=" * 60)

test_probs = final_model.predict(X_test)
test_pred = test_probs.argmax(axis=1)

test_ll = log_loss(y_test, test_probs, labels=list(range(num_classes)))
test_acc = accuracy_score(y_test, test_pred)
test_qwk = cohen_kappa_score(y_test, test_pred, weights="quadratic")

print(f"测试集 Log Loss: {test_ll:.4f}")
print(f"测试集 Accuracy: {test_acc:.4f}")
print(f"测试集 Quadratic Weighted Kappa: {test_qwk:.4f}")
print()

# ========== 步骤9: 特征重要性 ==========
print("=" * 60)
print("步骤9: 特征重要性")
print("=" * 60)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': final_model.feature_importance(importance_type='gain')
}).sort_values('importance', ascending=False)

print("Top 10 特征:")
print(feature_importance.head(10))
print()

# ========== 步骤10: 预测示例 ==========
print("=" * 60)
print("步骤10: 单个预测示例")
print("=" * 60)

# 展示几个具体的预测
sample_indices = [0, 1, 2]
for idx in sample_indices:
    real_idx = test_df.index[idx]
    resp_info = test_df.loc[real_idx, ['resp_age', 'resp_race', 'resp_party', 'candidate_id']]
    true_label = y_test.iloc[idx]
    pred_label = test_pred[idx]
    pred_probs = test_probs[idx]
    
    print(f"\n样本 {idx}:")
    print(f"  受访者: 年龄={resp_info['resp_age']}, 种族={resp_info['resp_race']}, 党派={resp_info['resp_party']}")
    print(f"  候选人: {resp_info['candidate_id']}")
    print(f"  真实标签: {true_label}")
    print(f"  预测标签: {pred_label}")
    print(f"  预测概率: {pred_probs.round(3)}")

print("\n" + "=" * 60)
print("测试完成！")
print("=" * 60)

