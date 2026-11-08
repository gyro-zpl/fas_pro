# ANES数据宽表转长表工具

这个工具用于将ANES（American National Election Studies）数据从宽表格式转换为长表格式，以便进行候选人支持度预测分析。

## 问题背景

在ANES数据集中：
- **宽表格式**：每个受访者一行，包含对多个候选人的支持度（多列）
- **问题**：不同年份的候选人不同，难以统一建模
- **解决方案**：转换为长表格式，每个受访者-候选人配对一行

## 数据结构

### 宽表格式（原始数据）

```
respondent_id | birthyr | gender | ... | ftobama | fttrump | fthrc | ...
--------------------------------------------------------------------------
1             | 1960    | 1      | ... | 100     | 1       | 76    | ...
2             | 1957    | 2      | ... | 39      | 28      | 52    | ...
```

### 长表格式（转换后）

```
respondent_id | year | candidate | support_score | birthyr | gender | ...
--------------------------------------------------------------------------
1             | 2016 | obama     | 100.0         | 1960    | 1      | ...
1             | 2016 | trump     | 1.0           | 1960    | 1      | ...
1             | 2016 | hrc       | 76.0          | 1960    | 1      | ...
2             | 2016 | obama     | 39.0          | 1957    | 2      | ...
2             | 2016 | trump     | 28.0          | 1957    | 2      | ...
```

## 数据列说明

### 受访者特征列
- `birthyr`: 出生年份
- `gender`: 性别
- `race`: 种族
- `educ`: 教育程度
- `marstat`: 婚姻状况
- `employ`: 就业状况
- `faminc`: 家庭收入
- `votereg`: 选民登记状态

### 党派意向列
- `pid7`: 党派认同（7点量表）
- `ideo5`: 意识形态（5点量表）
- `lcself`: 自由-保守自评

### 候选人支持度列（ft开头）
- `ftobama`, `fttrump`, `fthrc`, `ftcarson`, `ftrubio`, `ftcruz`, 等等
- 值范围：0-100（0=非常不支持，100=非常支持）
- 特殊值：
  - `997`: 拒绝回答（转换为缺失值）
  - `998`: 不知道（转换为缺失值）
  - `NaN`: 缺失值

## 使用方法

### 1. 单个文件转换

```bash
python anes_wide_to_long.py <输入文件> <输出文件> [年份]
```

**示例：**
```bash
# 转换2016年数据
python anes_wide_to_long.py \
    /home/cyyuan/FAS_PRO/data/anes/anes_pilot_2016.csv \
    /home/cyyuan/FAS_PRO/data/anes/anes_pilot_2016_long.csv \
    2016

# 如果文件名包含年份，可以省略年份参数（自动推断）
python anes_wide_to_long.py \
    anes_pilot_2016.csv \
    anes_pilot_2016_long.csv
```

### 2. 批量处理多个年份

编辑 `batch_process_anes.py` 中的文件路径：

```python
input_files = {
    2016: '/path/to/anes_pilot_2016.csv',
    2017: '/path/to/anes_pilot_2017.csv',
    2018: '/path/to/anes_pilot_2018.csv',
    2019: '/path/to/anes_pilot_2019.csv',
    2020: '/path/to/anes_pilot_2020.csv',
}
```

然后运行：

```bash
python batch_process_anes.py
```

这将：
1. 处理每个年份的数据，生成独立的长表文件
2. 合并所有年份为一个文件：`anes_pilot_all_years_long.csv`
3. 生成数据质量报告

### 3. 在Python代码中使用

```python
from anes_wide_to_long import wide_to_long_anes

# 转换单个文件
success = wide_to_long_anes(
    input_file='anes_pilot_2016.csv',
    output_file='anes_pilot_2016_long.csv',
    year=2016
)

# 读取转换后的数据
import pandas as pd
df_long = pd.read_csv('anes_pilot_2016_long.csv')

print(df_long.head())
```

## 输出数据格式

转换后的长表包含以下列：

| 列名 | 说明 |
|------|------|
| `respondent_id` | 受访者ID（原始ID） |
| `year` | 年份 |
| `candidate` | 候选人/对象名称（去掉ft前缀） |
| `support_score` | 支持度评分（0-100） |
| `is_missing` | 缺失标记（0=有效，1=缺失） |
| `birthyr` | 出生年份 |
| `gender` | 性别 |
| ... | 其他受访者特征列 |

## 数据处理细节

### 1. 候选人识别
- 自动识别所有`ft`开头的列作为候选人/对象支持度
- 去掉`ft`前缀作为候选人名称
- 注意：`ftblack`和`ftwhite`是种族群体的feeling thermometer，不是候选人

### 2. 缺失值处理
- 原始缺失值（NaN）保留
- 特殊值997、998标记为缺失，并设置`is_missing=1`
- 可以根据`is_missing`列过滤有效数据

### 3. 受访者ID
- 如果原始数据没有`respondent_id`，自动生成（使用行索引）
- 批量处理时，会生成`global_respondent_id`（格式：`year_id`）

## 示例：数据分析工作流

```python
import pandas as pd
import numpy as np

# 1. 读取长表数据
df = pd.read_csv('anes_pilot_all_years_long.csv')

# 2. 过滤有效数据（去除缺失值）
df_valid = df[df['is_missing'] == 0].copy()

# 3. 按候选人分组统计
candidate_stats = df_valid.groupby('candidate').agg({
    'support_score': ['mean', 'std', 'count']
})
print(candidate_stats)

# 4. 按年份和候选人分组
year_candidate_stats = df_valid.groupby(['year', 'candidate']).agg({
    'support_score': 'mean'
}).unstack()
print(year_candidate_stats)

# 5. 构建机器学习特征
# 特征：受访者特征 + 党派意向
feature_cols = ['birthyr', 'gender', 'race', 'educ', 'marstat', 
                'employ', 'faminc', 'votereg', 'pid7', 'ideo5', 'lcself']
X = df_valid[feature_cols]
y = df_valid['support_score']

# 6. 添加候选人编码
from sklearn.preprocessing import LabelEncoder
df_valid['candidate_encoded'] = LabelEncoder().fit_transform(df_valid['candidate'])
```

## 常见问题

### Q1: 如何区分候选人和其他对象（如种族群体）？

A: 可以手动创建候选人列表，只保留真实候选人：

```python
# 真实候选人列表（需要根据实际情况调整）
real_candidates = ['obama', 'trump', 'hrc', 'carson', 'rubio', 
                   'cruz', 'sanders', 'fiorina', 'jeb', 'biden', 
                   'warren', 'buttigieg', 'harris', 'klobuchar']

# 过滤只包含真实候选人的数据
df_candidates = df[df['candidate'].isin(real_candidates)]
```

### Q2: 不同年份的列名不一致怎么办？

A: 脚本会自动识别所有存在的列。如果某些年份缺少某些特征列，可以：

```python
# 合并后统一处理缺失列
all_feature_cols = ['birthyr', 'gender', 'race', 'educ', 'marstat', 
                    'employ', 'faminc', 'votereg', 'pid7', 'ideo5', 'lcself']

# 确保所有列都存在，缺失的填充NaN
for col in all_feature_cols:
    if col not in df.columns:
        df[col] = np.nan
```

### Q3: 如何处理年份特异的候选人？

A: 这正是长表格式的优势！每个候选人只在出现的年份有记录：

```python
# 查看每个候选人出现的年份
candidate_years = df.groupby('candidate')['year'].unique()
for candidate, years in candidate_years.items():
    print(f"{candidate}: {years}")
```

## 文件结构

```
feature_engineer/
├── anes_wide_to_long.py       # 核心转换脚本
├── batch_process_anes.py      # 批量处理脚本
├── README_anes_w2l.md         # 本文档
└── w2l.py                     # 原始参考脚本（不同格式）
```

## 性能说明

- **数据量**：1200受访者 × 11候选人 = 13,200行（单年）
- **处理时间**：通常<1秒/年
- **内存占用**：很小（<100MB for 5年数据）

## 更新日志

- **2025-10-16**: 初始版本，支持ANES pilot数据2016-2020

## 联系方式

如有问题或建议，请联系项目维护者。

