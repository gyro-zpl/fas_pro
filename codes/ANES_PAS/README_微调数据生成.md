# ANES选民支持度预测 - 微调数据集生成器

## 概述

本工具基于你的**特征说明文档**中描述的70+个交互特征，为ANES选民支持度预测任务生成高质量的LLM微调数据集。

## 核心设计理念

### 📊 数据特征理解

根据你的特征文档，我们重点关注：

1. **最重要特征** (根据政治科学文献)：
   - `party_match`: 党派匹配（最强预测因子）
   - `ideology_distance_7`: 意识形态距离
   - `party_strength_resp`: 党派认同强度
   - `demographic_similarity`: 综合相似度

2. **交互效应**：
   - 党派 × 意识形态
   - 人口统计 × 党派
   - 年龄差异 × 代际效应
   - 现任优势效应

### 🎯 9大采样策略

| 策略 | 说明 | 特点 | 样本数 |
|------|------|------|--------|
| `balanced` | 平衡分层采样 | 覆盖所有支持度范围 | 500 |
| `hard_cases` | 难例挖掘 | 中间态度、边界情况 | 300 |
| `temporal` | 时间多样性 | 不同年份、选举周期 | 300 |
| `extreme` | 极端案例 | 强烈支持/反对 | 200 |
| `party_ideology` | 党派×意识形态组合 | **核心！** 5种关键组合 | 600 |
| `demographic_sim` | 综合相似度分层 | 利用similarity指标 | 400 |
| `incumbent` | 现任优势效应 | 现任vs非现任对比 | 300 |
| `cross_pressure` | 交叉压力情况 | **难点！** 冲突的线索 | 300 |
| `candidate` | 候选人特定 | 每个候选人均衡 | 150×N |

**默认策略集**（推荐）：`balanced`, `party_ideology`, `demographic_sim`, `cross_pressure`, `extreme`

## 使用方法

### 1. 基础使用（使用默认策略）

```bash
python generate_finetune_data.py
```

这将：
- 使用默认5个核心策略
- 输出到 `/home/cyyuan/FAS_PRO/data/anes/finetune/anes_finetune_dataset.jsonl`
- 生成约 2000-2500 个样本

### 2. 使用所有策略

```bash
python generate_finetune_data.py --all
```

生成约 3000-3500 个样本，覆盖所有9个策略。

### 3. 自定义策略

```bash
python generate_finetune_data.py \
  --strategies party_ideology cross_pressure incumbent \
  --output /path/to/output.jsonl
```

### 4. 控制样本数量

```bash
# 生成后随机下采样到1000个
python generate_finetune_data.py --all --sample-size 1000
```

### 5. 查看更多示例

```bash
python generate_finetune_data.py --show-examples 5
```

## 输出格式

### JSONL格式（每行一个JSON）

```json
{
  "instruction": "Predict the support score (0-100) for this voter-candidate pair.\n\nVoter Profile:\n- Demographics: 45-year-old Female White\n- Education: Bachelor's degree\n- Political: Democrat, Liberal\n\nCandidate: Joe Biden\n- Party: Democrat\n- Position: President  \n- Age: 78\n\nKey Interactions:\n- Party alignment: Same party\n- Ideology distance: 0.5 (on 7-point scale)\n- Age difference: 33 years\n- Year: 2020\n\nSupport score (0-100):",
  "input": "",
  "output": "85",
  "metadata": {
    "strategy": "party_ideology_focused",
    "combination": "perfect_align",
    "true_score": 85.0
  }
}
```

### 三种Prompt级别

1. **`minimal`**: 只有基本信息（年龄、党派、意识形态）
2. **`core`** (推荐): 包含关键交互特征（party_match, ideology_distance等）
3. **`full`**: 包含所有特征（demographic_similarity, incumbent等）

## 策略详解

### 🔥 核心策略：`party_ideology` (党派×意识形态)

根据你的特征文档，这两个特征是**最重要的预测因子**。该策略采样：

1. **完美对齐** (`perfect_align`): 同党派 + 同意识形态侧
   - 期望：高支持度 (70-95)
   
2. **同党异意** (`same_party_diff_ideo`): 同党派但意识形态距离大
   - 考验：党派忠诚 vs 意识形态分歧
   
3. **异党同意** (`diff_party_same_ideo`): 不同党派但意识形态接近
   - 考验：意识形态认同 vs 党派偏见
   
4. **完全不匹配** (`total_mismatch`): 党派和意识形态都不同
   - 期望：低支持度 (5-30)
   
5. **独立选民** (`independent`): party_match == -1
   - 特殊群体，最难预测

### 🧩 难点策略：`cross_pressure` (交叉压力)

专门采样那些**信号冲突**的困难案例：

- **党派忠诚 vs 意识形态分歧**: 同党但意识形态差3+
- **意识形态认同 vs 党派忠诚**: 异党但意识形态距离≤1  
- **身份政治 vs 党派**: demographic_similarity ≥ 0.7 但党派不同

这些是模型**最容易出错**的情况！

### 🏛️ 情境策略：`incumbent` (现任优势)

美国政治中的**现任优势效应**（incumbency advantage）：

- 现任总统/副总统通常有更高基准支持度
- 采样现任 vs 非现任，让模型学会这种模式

### 📈 综合策略：`demographic_sim` (综合相似度)

利用你的特征工程中的 `demographic_similarity` 综合指标：

- 该指标整合了种族、性别、年龄、党派、意识形态的匹配度
- 按相似度分5档采样：very_low (0-0.2) 到 very_high (0.8-1.0)

## 数据质量保证

### ✅ 去重

自动移除prompt完全相同的样本。

### ✅ 打乱

最终数据集会随机打乱顺序，避免策略聚集。

### ✅ 元数据

每个样本都包含：
- `strategy`: 采样策略
- `true_score`: 真实支持度（用于后续评估）
- 其他策略特定信息

## 预期效果

### 训练后模型应该学会：

1. **党派极化规律**
   - 同党派 → 高支持 (通常 60-90)
   - 异党派 → 低支持 (通常 10-40)
   - 独立选民 → 中间态度 (30-60)

2. **意识形态微调**
   - 即使同党派，意识形态距离大也会降低支持度
   - 异党派但意识形态接近，可能有一定支持度

3. **人口统计效应**
   - 同种族/同性别/同代际 → 轻微加分
   - 但远不如党派重要

4. **现任优势**
   - 现任总统/副总统有基准加成

5. **复杂权衡**
   - 当多个线索冲突时，如何权衡
   - 党派 > 意识形态 > 人口统计

## 后续使用

### 微调格式转换

根据你使用的框架，可能需要转换格式：

#### OpenAI格式

```python
import json

with open('anes_finetune_dataset.jsonl', 'r') as f:
    data = [json.loads(line) for line in f]

# 转换为OpenAI格式
openai_format = []
for sample in data:
    openai_format.append({
        "messages": [
            {"role": "system", "content": "You are a political science expert predicting voter support."},
            {"role": "user", "content": sample['instruction']},
            {"role": "assistant", "content": sample['output']}
        ]
    })

with open('anes_openai_format.jsonl', 'w') as f:
    for item in openai_format:
        f.write(json.dumps(item) + '\n')
```

#### LLaMA/Alpaca格式

已经是标准的 instruction-input-output 格式，可直接使用！

### 评估基准

保留一部分测试集（如2022年数据），用于评估：

```python
# 计算MAE (Mean Absolute Error)
mae = np.mean(np.abs(predictions - true_scores))

# 计算分类准确率（5个等级）
accuracy = (predicted_bins == true_bins).mean()

# 按难度分析
hard_cases = df[df['cross_pressure']]
mae_hard = np.mean(np.abs(predictions[hard_cases] - true_scores[hard_cases]))
```

## 高级技巧

### 1. 渐进式微调

```bash
# 阶段1: 只用简单样本（明确党派信号）
python generate_finetune_data.py --strategies party_ideology balanced

# 阶段2: 加入难例
python generate_finetune_data.py --strategies cross_pressure hard_cases

# 阶段3: 全部策略
python generate_finetune_data.py --all
```

### 2. 候选人特定模型

```bash
# 只为特定候选人生成数据
python generate_finetune_data.py --strategies candidate
```

### 3. 数据增强

可以在同一个选民-候选人对上使用不同的prompt级别：

```python
generator = ANESFinetuneDataGenerator(data_path)
row = df.iloc[0]

# 生成3个不同详细程度的prompt
for level in ['minimal', 'core', 'full']:
    prompt = generator._create_prompt_from_row(row, include_features=level)
    # ...
```

## 常见问题

### Q: 为什么输出只是一个数字？

A: 这是回归预测任务。简洁的输出（如"85"）比冗长的解释更适合：
- 减少模型学习负担
- 避免幻觉（hallucination）
- 便于后处理和评估

如果你想要解释，可以改为分步prompt：
```
Step 1: Analyze the key factors.
Step 2: Predict the score.
```

### Q: 样本数量够吗？

A: 取决于模型大小：
- **小模型** (7B参数): 1000-2000样本足够
- **中等模型** (13B): 2000-3000样本
- **大模型** (70B+): 可能需要更多

你的数据有12万条记录，所以可以生成更多！

### Q: 如何处理缺失值？

A: 特征工程脚本已经处理过了：
- 数值型：用中位数填充
- 分类型：标记为'unknown'
- 你的数据中 `demographic_similarity` 等综合指标已经妥善处理

## 文件清单

```
/home/cyyuan/FAS_PRO/codes/ANES_PAS/
├── generate_finetune_data.py          # 主脚本
├── README_微调数据生成.md              # 本文档
└── feature_engineer/
    └── 特征说明文档.md                 # 特征详解

/home/cyyuan/FAS_PRO/data/anes/
├── processed/
│   ├── anes_all_years_with_features.csv  # 输入数据
│   ├── X_features_classification.csv
│   ├── y_target_classification.csv
│   └── feature_summary.csv
└── finetune/
    └── anes_finetune_dataset.jsonl       # 输出数据
```

## 总结

本工具充分利用了你精心设计的**70+个交互特征**，通过9大策略生成多样化、高质量的微调数据，特别关注：

✅ **党派×意识形态** 这一核心组合  
✅ **交叉压力** 这一难点情况  
✅ **综合相似度** 这一整合指标  
✅ **现任优势** 这一情境效应  

期待这能显著提升你的模型在选民支持度预测任务上的表现！🚀

---

**作者**: AI Assistant  
**版本**: 1.0  
**日期**: 2025-11-10

