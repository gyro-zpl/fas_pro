# ANES 宽表转长表建模方案说明

## 📌 方案概述

本方案用于解决**跨年份、跨候选人的支持度预测问题**。核心思路是将每年的宽表数据（一个受访者对多个候选人的支持度）转换为长表格式（一行 = 一个受访者-候选人对），从而学习"受访者特征 × 候选人特征 × 年份情境 → 支持等级"的通用映射规律。

### 问题场景

- **原始数据格式**：每年的候选人不同
  - 2012: [受访者信息, 对Trump支持度, 对Obama支持度, 对Clinton支持度]
  - 2014: [受访者信息, 对Rubio支持度, 对Harris支持度, 对Jackson支持度]
  - 2016: [受访者信息, 对Biden支持度, 对Trump支持度, 对Jason支持度]

- **目标**：用历史年份数据训练模型，预测新年份中新候选人的支持度

### 方案优势

✅ **跨候选人泛化**：模型学习的是通用规律而非特定候选人的记忆  
✅ **数据高效利用**：每个受访者-候选人对都是一个训练样本  
✅ **特征可扩展**：可灵活添加候选人元数据、宏观经济指标等  
✅ **迁移学习**：历史候选人的规律可迁移到新候选人  

---

## 🔧 核心模块

### 1. `wide_to_long()` - 宽表转长表

```python
def wide_to_long(df_wide, year, candidate_map, scale_type='text'):
    """
    将某年的宽表转换为长表
    
    Args:
        df_wide: 宽表数据，包含 support_xxx 列
        year: 年份
        candidate_map: 候选人名称映射，如 {"trump": "Trump"}
        scale_type: 支持度量表类型 ('text', '1-6', '0-5')
    
    Returns:
        长表数据，每行是一个 (受访者, 候选人, 支持度)
    """
```

**关键改进**：
- 自动识别 `support_` 开头的列
- 统一支持度量表到 0-5 范围
- 添加缺失值指示器
- 避免 `year` 列重复问题

### 2. `build_interactions()` - 构造交互特征

```python
def build_interactions(df):
    """构造受访者-候选人交互特征"""
    # 意识形态距离
    df["ideology_distance"] = |受访者意识形态 - 候选人意识形态|
    # 党派一致性
    df["party_alignment"] = (受访者党派 == 候选人党派)
    # 年龄代沟
    df["age_gap"] = |受访者年龄 - 候选人年龄|
    # 种族一致性
    df["race_alignment"] = (受访者种族 == 候选人种族)
```

**重要性排名**（从测试结果）：
1. `ideology_distance` (65.6%) - **最重要**
2. `party_alignment` (15.6%)
3. 其他特征

### 3. `prepare_candidate_metadata()` - 候选人元数据

```python
candidate_metadata = [
    {"candidate_id": "Trump", "cand_party": "Republican", "cand_age": 66, ...},
    {"candidate_id": "Obama", "cand_party": "Democrat", "cand_age": 51, ...},
    ...
]
candidate_meta = prepare_candidate_metadata(candidate_metadata)
```

**必需的候选人特征**：
- `cand_party`：党派
- `cand_age`：年龄
- `cand_race`：种族
- `cand_ideology_score`：意识形态得分（1-10）
- （可选）从政经验、现任状态、民调支持率等

---

## 📊 完整流程

### 步骤1: 准备数据

```python
# 创建每年的宽表
df_2012 = pd.DataFrame({
    'resp_age': [...],
    'resp_race': [...],
    'resp_party': [...],
    'support_trump': ['support', 'unsupport', ...],
    'support_obama': ['neutral', 'support', ...],
    ...
})
```

### 步骤2: 宽表转长表

```python
from w2l import wide_to_long

# 定义候选人映射
candidate_map = {"trump": "Trump", "obama": "Obama", ...}

# 转换为长表
long_2012 = wide_to_long(df_2012, year=2012, candidate_map=candidate_map)
long_2014 = wide_to_long(df_2014, year=2014, candidate_map=candidate_map_2014)

# 合并所有年份
all_long = pd.concat([long_2012, long_2014, ...], ignore_index=True)
```

**输出格式**：
```
| respondent_id | resp_age | resp_race | resp_party | candidate_id | year | target |
|---------------|----------|-----------|------------|--------------|------|--------|
| 0             | 25       | white     | Republican | Trump        | 2012 | 4      |
| 0             | 25       | white     | Republican | Obama        | 2012 | 1      |
| 0             | 25       | white     | Republican | Clinton      | 2012 | 2      |
| 1             | 30       | black     | Democrat   | Trump        | 2012 | 1      |
...
```

### 步骤3: 合并候选人元数据

```python
all_long = all_long.merge(candidate_meta, on="candidate_id", how="left")
```

### 步骤4: 构造交互特征

```python
from w2l import build_interactions

all_long = build_interactions(all_long)
```

### 步骤5: 划分训练/测试集

```python
# 训练集：历史年份
train_df = all_long[all_long.year < 2016].copy()
# 测试集：目标年份
test_df = all_long[all_long.year == 2016].copy()
```

### 步骤6: 特征工程

```python
from sklearn.preprocessing import OrdinalEncoder

# 选择特征
exclude = ["respondent_id", "candidate_id", "year", "target", "target_raw"]
feature_cols = [c for c in train_df.columns if c not in exclude]

# 数值特征：填充中位数
num_cols = train_df[feature_cols].select_dtypes(include=[np.number]).columns
for c in num_cols:
    med = train_df[c].median()
    train_df[c].fillna(med, inplace=True)
    test_df[c].fillna(med, inplace=True)

# 类别特征：编码
cat_cols = [c for c in feature_cols if c not in num_cols]
enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
train_df[cat_cols] = enc.fit_transform(train_df[cat_cols].fillna("MISSING"))
test_df[cat_cols] = enc.transform(test_df[cat_cols].fillna("MISSING"))
```

### 步骤7: 模型训练

```python
import lightgbm as lgb
from sklearn.model_selection import LeaveOneGroupOut

# 按年份分组的交叉验证
logo = LeaveOneGroupOut()
groups = train_df["year"].values

params = {
    "objective": "multiclass",
    "num_class": 6,  # 0-5共6个等级
    "metric": "multi_logloss",
    "learning_rate": 0.05,
    "num_leaves": 31,
    ...
}

for fold, (tr_idx, val_idx) in enumerate(logo.split(X_train, y_train, groups)):
    # 训练模型
    model = lgb.train(params, train_set, ...)
    # 验证
    ...
```

### 步骤8: 预测与评估

```python
from sklearn.metrics import log_loss, accuracy_score, cohen_kappa_score

# 预测
test_probs = model.predict(X_test)
test_pred = test_probs.argmax(axis=1)

# 评估
print("Log Loss:", log_loss(y_test, test_probs))
print("Accuracy:", accuracy_score(y_test, test_pred))
print("Quadratic Weighted Kappa:", cohen_kappa_score(y_test, test_pred, weights="quadratic"))
```

---

## 🧪 测试结果

运行 `python test_w2l.py` 后的输出：

```
训练集: 60样本 (2012-2014年, 6个候选人)
测试集: 30样本 (2016年, 3个候选人)

交叉验证:
  Fold 0 (Year 2012) - logloss=1.117 | acc=0.300 | qwk=0.000
  Fold 1 (Year 2014) - logloss=1.159 | acc=0.300 | qwk=0.000

测试集评估 (2016年):
  Log Loss: 1.036
  Accuracy: 0.600  ← 比交叉验证提升
  Quadratic Weighted Kappa: 0.589

特征重要性:
  1. ideology_distance: 65.6%  ← 意识形态距离是关键
  2. party_alignment: 15.6%    ← 党派一致性次之
```

**关键发现**：
- ✅ 模型成功在新候选人上泛化（测试集包含训练集没见过的候选人）
- ✅ 意识形态距离是最重要的特征
- ⚠️ 样本量较小导致交叉验证性能波动

---

## ⚠️ 注意事项

### 1. 数据泄漏风险

**问题**：如果使用当年的宏观变量预测当年的支持度  
**解决**：使用滞后值或移动平均

```python
# ❌ 错误：使用2016年数据预测2016年
macro_2016 = {"year": 2016, "unemployment": 4.9}

# ✅ 正确：使用上一年的数据
macro_2016 = {"year": 2016, "unemployment_lag1": 5.3}  # 2015年的值
```

### 2. 交叉验证策略

**推荐**：按年份 Leave-One-Group-Out  
**目的**：模拟真实场景（用过去年份预测未来年份）

```python
from sklearn.model_selection import LeaveOneGroupOut
logo = LeaveOneGroupOut()
groups = train_df["year"].values  # 关键：按年份分组
```

### 3. 类别数不匹配

**问题**：某些fold中缺少某些支持度等级  
**解决**：在 `log_loss` 中显式指定所有标签

```python
# ✅ 正确
log_loss(y_val, y_pred_proba, labels=list(range(num_classes)))
```

### 4. 候选人特征的重要性

**缺失候选人特征 → 模型无法泛化到新候选人**

必需特征：
- 党派（最重要）
- 意识形态得分（最重要）
- 年龄
- 种族
- 性别
- 从政经验

### 5. 支持度量表标准化

不同年份可能使用不同量表：
- 2012: 1-6
- 2016: 0-5
- 2020: 文本 ('support', 'neutral', 'unsupport')

**解决**：使用 `standardize_support_level()` 统一到 0-5

---

## 📈 进阶优化

### 1. 添加时序特征

```python
# 距离选举日天数
df["days_to_election"] = (election_date - survey_date).days

# 竞选阶段
df["campaign_stage"] = "primary" / "general"
```

### 2. 添加竞争环境特征

```python
# 同年其他候选人的平均支持度
df["other_candidates_avg_support"] = ...

# 该候选人的相对排名
df["candidate_rank"] = ...
```

### 3. 多任务学习

同时预测：
- 支持度等级
- 是否投票给该候选人
- 是否捐款

### 4. 候选人Embedding

使用神经网络学习候选人的低维表示：

```python
import torch.nn as nn

class CandidateEmbedding(nn.Module):
    def __init__(self, num_candidates, embed_dim=8):
        super().__init__()
        self.embedding = nn.Embedding(num_candidates, embed_dim)
```

---

## 📁 文件说明

- `w2l.py`：核心功能模块
  - `wide_to_long()`：宽表转长表
  - `build_interactions()`：构造交互特征
  - `prepare_candidate_metadata()`：候选人元数据
  - `standardize_support_level()`：支持度标准化

- `test_w2l.py`：完整测试样例
  - 演示完整流程
  - 包含模拟数据
  - 可直接运行

- `README_w2l.md`：本文档

---

## 🚀 快速开始

```bash
# 1. 安装依赖
pip install pandas numpy scikit-learn lightgbm

# 2. 运行测试
cd /home/cyyuan/FAS_PRO/codes/ANES_PAS/feature_engineer
python test_w2l.py

# 3. 在自己的数据上使用
from w2l import wide_to_long, build_interactions

# 转换数据
long_df = wide_to_long(your_wide_df, year=2020, candidate_map={...})

# 添加候选人特征和交互
long_df = long_df.merge(candidate_meta, on="candidate_id")
long_df = build_interactions(long_df)

# 训练模型
...
```

---

## 📚 理论基础

本方案借鉴了：

1. **Panel Data Analysis**：面板数据回归
2. **Transfer Learning**：跨域知识迁移
3. **Collaborative Filtering**：类似推荐系统的用户-物品建模
4. **Political Science**：投票行为理论
   - Proximity Theory：选民倾向于支持意识形态相近的候选人
   - Party Identification：党派认同是最强预测因子

---

## ❓ 常见问题

**Q: 为什么不直接用候选人名字作为特征？**  
A: 候选人名字对新候选人无法泛化。必须使用候选人的**属性特征**（党派、意识形态等）。

**Q: 如果某候选人在训练集和测试集都出现（如Trump）怎么办？**  
A: 这是合理的。模型会同时学习：
   - 该候选人的特定模式
   - 一般性的规律（通过其他候选人）

**Q: 交叉验证为什么要按年份分组？**  
A: 因为真实场景是用过去年份预测未来年份。按候选人分组无法验证时序泛化能力。

**Q: 数据量太小怎么办？**  
A: 
   - 增加更多历史年份
   - 使用正则化防止过拟合
   - 尝试更简单的模型（逻辑回归）

---

## 📝 引用

如果你的研究使用了本方案，请引用：

```
ANES Panel Data Wide-to-Long Modeling Framework
Author: [Your Name]
Year: 2025
Repository: [URL]
```

---

## 🤝 贡献

欢迎提交Issue和PR！

特别感谢：
- ANES (American National Election Studies) 提供数据
- LightGBM团队
- scikit-learn团队

