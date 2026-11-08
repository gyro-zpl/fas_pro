# 受访者×候选人交互特征生成器 (v2.0)

## 概述

这个脚本用于创建受访者和候选人之间的交互特征，支持多年份数据处理，灵活应对不同年份的属性差异。

## 主要特点

### 1. 灵活的列名处理
- **自动标准化列名**：处理不同年份的命名差异（如 `race7` → `race`, `marital1` → `marstat`）
- **智能合并**：避免列名冲突，自动添加 `_resp` 和 `_cand` 后缀

### 2. 丰富的交互特征

#### 年龄特征 (8个)
- `age_resp`, `age_cand`: 受访者和候选人年龄
- `age_difference`: 年龄差异（绝对值）
- `age_difference_ratio`: 年龄差异比例
- `age_gap_category`: 年龄代差分类（同代/相邻代/跨代）
- `resp_older`: 受访者是否年长
- `age_decade_resp`, `age_decade_cand`: 年龄段（按10年）
- `age_decade_diff`: 年龄段差异

#### 党派特征 (9个)
- `party_alignment`: 受访者党派倾向（1=民主党，2=共和党，0=独立）
- `party_strength_resp`: 党派强度（距离中间派的距离）
- `party_match`: 党派匹配度（-1=独立选民，0=不匹配，1=匹配）
- `same_party`: 同党派二值特征
- `opposite_party`: 相反党派
- `pid7_cand`: 候选人党派映射到pid7量表
- `party_distance`: 党派距离
- `party_strength_x_match`: 党派强度×匹配度
- `party_strength_x_distance`: 党派强度×距离

#### 意识形态特征 (最多9个)
- `ideology_cand_5scale`: 候选人意识形态（5分制）
- `ideology_distance_5`: 意识形态差异（5分制）
- `ideology_distance_7`: 意识形态差异（7分制）
- `ideology_same_side_5`, `ideology_same_side_7`: 同一阵营
- `both_extreme_ideology`: 都在极端
- `both_moderate_ideology`: 都在中间
- `ideology_strength_resp`, `ideology_strength_cand`: 意识形态强度

#### 人口统计特征 (8个)
- **种族**: `race_match`, `both_white`, `both_minority`, `cross_race`
- **性别**: `gender_match`, `female_resp_male_cand`, `male_resp_female_cand`, `same_gender`

#### 复合交互特征 (10个)
- `party_match_x_ideology_dist`: 党派匹配×意识形态距离
- `party_same_x_ideology_same`: 同党派×同意识形态
- `race_match_x_party_match`: 种族匹配×党派匹配
- `gender_match_x_party_match`: 性别匹配×党派匹配
- `age_diff_x_educ`: 年龄差异×教育程度
- `age_diff_x_party_dist`: 年龄差异×党派距离
- `educ_x_ideology_dist`: 教育×意识形态距离
- `income_x_party_dist`: 收入×党派距离
- `party_ideology_race_align`: 党派×意识形态×种族（三阶）
- `demographic_similarity`: 综合人口统计相似度得分

#### 情境特征 (6个)
- `year`, `year_normalized`: 年份及归一化
- `is_election_year`: 选举年标记
- `is_presidential_year`: 总统选举年标记
- `position_encoded`: 候选人职位编码
- `is_incumbent`: 是否现任
- `is_candidate`: 是否候选人

### 3. 完善的数据处理
- **缺失值处理**：数值型用中位数填充，分类型用众数或'unknown'
- **类型转换**：自动转换数值型列
- **质量检查**：详细的数据质量报告
- **异常处理**：健壮的错误处理和日志记录

## 使用方法

### 基本用法

```python
from create_interaction_features import main

# 使用默认配置（自动检测年份）
df_all, X, y, feature_names = main()
```

### 自定义配置

```python
from create_interaction_features import main

# 指定年份和路径
df_all, X, y, feature_names = main(
    data_dir='/home/cyyuan/FAS_PRO/data/anes',
    years=[2016, 2018, 2019, 2020],
    output_dir='/home/cyyuan/FAS_PRO/data/anes/processed',
    target_type='classification'  # 或 'regression'
)
```

### 单独使用特征生成函数

```python
import pandas as pd
from create_interaction_features import (
    load_candidate_metadata, 
    create_interaction_features,
    prepare_training_data
)

# 1. 加载候选人元数据
df_meta = load_candidate_metadata('candidate_metadata.csv')

# 2. 加载长表数据
df_long = pd.read_csv('anes_pilot_2020_long.csv')

# 3. 创建交互特征
df_features = create_interaction_features(df_long, df_meta)

# 4. 准备训练数据
X, y, feature_names, df_valid = prepare_training_data(
    df_features, 
    target_type='classification'
)
```

## 输入数据格式

### 长表数据 (`anes_pilot_{year}_long.csv`)

必需列：
- `respondent_id`: 受访者ID
- `year`: 年份
- `candidate`: 候选人名称
- `support_score`: 支持度分数（0-100）
- `is_missing`: 是否缺失（0=有效，1=缺失）

可选列（受访者特征）：
- `birthyr` 或 `age`: 出生年份或年龄
- `gender`: 性别（1=男，2=女）
- `race` / `race7` / `race5`: 种族
- `educ`: 教育程度
- `marstat` / `marital1`: 婚姻状况
- `employ`: 就业状况
- `faminc` / `income`: 家庭收入
- `votereg`: 选民登记
- `pid7`: 党派认同（1-7量表）
- `ideo5`: 意识形态（5分制）
- `lcself`: 意识形态（7分制）

### 候选人元数据 (`candidate_metadata.csv`)

必需列：
- `candidate`: 候选人名称
- `year`: 年份
- `party`: 党派（1=民主党，2=共和党）
- `ideology`: 意识形态（1-7，1=极端自由派，7=极端保守派）

可选列：
- `birthyr`: 出生年份
- `race`: 种族
- `gender`: 性别
- `position`: 职位（president, candidate, senator等）

## 输出文件

运行脚本后，会在 `output_dir` 下生成以下文件：

### 1. 单年特征文件
```
anes_2016_with_features.csv
anes_2018_with_features.csv
anes_2019_with_features.csv
anes_2020_with_features.csv
```

### 2. 合并数据
```
anes_all_years_with_features.csv  # 所有年份合并，包含所有特征
```

### 3. 训练数据
```
X_features_classification.csv     # 特征矩阵（分类任务）
y_target_classification.csv       # 目标变量（分类任务）
X_features_regression.csv         # 特征矩阵（回归任务）
y_target_regression.csv           # 目标变量（回归任务）
feature_summary.csv               # 特征摘要（包含特征分类）
```

## 特征分类

所有特征被分为以下类别：

1. **respondent_basic**: 受访者基础特征（年龄、性别、种族、教育等）
2. **respondent_political**: 受访者政治倾向（党派、意识形态）
3. **candidate_basic**: 候选人基础特征（党派、意识形态、年龄等）
4. **candidate_contextual**: 候选人情境特征（职位、是否现任等）
5. **age_interaction**: 年龄交互特征
6. **party_interaction**: 党派交互特征
7. **ideology_interaction**: 意识形态交互特征
8. **demographic_interaction**: 人口统计交互特征
9. **compound_interaction**: 复合交互特征（二阶、三阶）
10. **year_contextual**: 年份情境特征

## 目标变量

### 分类任务 (classification)
将 `support_score` (0-100) 离散化为5个等级：
- 0: 强烈反对 (0-20)
- 1: 反对 (21-40)
- 2: 中立 (41-60)
- 3: 支持 (61-80)
- 4: 强烈支持 (81-100)

### 回归任务 (regression)
直接使用原始 `support_score` (0-100)

## 特点与优势

### 相比原版本的改进

1. **更灵活的数据处理**
   - 自动适配不同年份的列名差异
   - 支持部分特征缺失
   - 更健壮的错误处理

2. **更丰富的特征工程**
   - 从原来的约20个特征增加到50+个特征
   - 新增复合交互特征（二阶、三阶）
   - 新增综合相似度得分

3. **更好的代码组织**
   - 模块化的特征计算函数
   - 清晰的特征分类
   - 详细的日志和进度报告

4. **更完善的输出**
   - 特征摘要文件
   - 详细的数据质量报告
   - 处理摘要统计

## 应用场景

### 预测候选人支持率

模型学习的映射关系：
```
受访者特征 × 候选人特征 × 年份情境 → 支持等级
```

关键点：
- **通用规律学习**：模型学习的是一般性的政治态度规律，而非特定候选人的记忆
- **样本设计**：每个"受访者×候选人"组合都是一个独立样本
- **迁移学习**：通过历史数据（如2016-2020）训练，预测未来年份（如2024）

### 示例工作流

```python
# 1. 生成交互特征
df_all, X, y, feature_names = main(
    years=[2016, 2018, 2019, 2020],
    target_type='classification'
)

# 2. 训练模型（LightGBM示例 - 后续步骤）
import lightgbm as lgb
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = lgb.LGBMClassifier(
    objective='multiclass',
    num_class=5,
    learning_rate=0.05,
    n_estimators=500
)

model.fit(X_train, y_train)

# 3. 预测新年份（如2024）
# 准备2024年的数据，使用相同的特征工程流程
# df_2024_features = create_interaction_features(df_2024_long, df_meta)
# predictions = model.predict(X_2024)
```

## 注意事项

1. **候选人元数据**：确保候选人元数据覆盖所有年份和候选人
2. **列名一致性**：虽然脚本会自动标准化，但保持一致性可以减少潜在问题
3. **缺失值**：某些年份缺少某些特征时，相关交互特征会被自动跳过
4. **内存使用**：处理大量数据时可能占用较多内存

## 依赖包

```bash
pip install pandas numpy
```

## 版本历史

- **v2.0** (2025-10-16)
  - 完全重构代码结构
  - 新增列名标准化功能
  - 大幅扩展交互特征（50+个）
  - 改进错误处理和日志
  - 添加特征分类和摘要

- **v1.0** (初始版本)
  - 基本交互特征生成
  - 简单的数据合并和处理

## 作者

CY Yuan

## 许可

MIT License

