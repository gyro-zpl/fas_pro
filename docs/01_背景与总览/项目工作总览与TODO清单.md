# FAS_PRO 项目工作总览与 TODO 清单

> **文档目的**：项目停摆约一年后的上下文恢复文档。汇总"基础工作 + 毕设工作"的定位、方法、代码、数据、实验结果，并给出已完成清单与待办清单。
>
> **编写日期**：2026-11-04
> **项目根目录**：`/home/jpzhao_team/cyyuan/FAS_PRO`
> **说明**：文中标注 `[已验证]` 的结论来自本地数据/结果文件；`[论文声称]` 来自 PDF 原文；`[文档声称]` 来自仓库内 md 文档，尚未独立复核。

---

## 0. 三十秒速览

- **基础工作**：LLM-S3 基准，提出 **PAS**（部分属性模拟）与 **FAS**（全属性模拟）两种"LLM 作为虚拟受访者"的评测范式，覆盖 **11 个真实调查数据集 × 4 个社会学领域**，只做提示（prompting）评测。
- **毕设工作**：SCSP 管道，把 PAS/FAS 从"评测"推进为"训练 + 机制分析"。在 **ANES** 上做个体级 PAS（长表 + 交互特征 + LoRA 微调），在 **GSS** 上做群体级 FAS（多模型外推 + IPF 联合分布 + 五类任务微调）。
- **代码现状**：数据生成、微调、推理、评测、量化部署的脚本基本齐全；实验跑过多轮；但**论文只写了一半**，且几个关键数字与本地结果对不上。
- **最大科学问题**：微调模型在 **2024 年数据上显著劣于 base 模型**（MAE 37.6 vs 23.7），根因是时间分布偏移 + 交叉压力泛化失败（已有专门诊断报告）。
- **最大工程问题**：全部脚本硬编码 `/home/cyyuan`，当前实际路径是 `/home/jpzhao_team/cyyuan`，直接跑会报错（共 255 处、64 个文件）。

---

## 1. 两篇文档的定位与关系

| | 基础工作 | 毕设工作 |
|---|---|---|
| 文件 | `docs/01_背景与总览/基础工作_LLM-S3_Virtual_Survey_Respondents.pdf` | `docs/04_论文稿件/毕设初稿_From_Prompting_to_Social_Cognition.pdf` |
| 标题 | Large Language Models as Virtual Survey Respondents: Evaluating Sociodemographic Response Generation | From Prompting to Social Cognition: Mechanistic Deepening of PAS and FAS Frameworks in Large Language Models |
| 性质 | 评测基准（benchmark） | 方法论文（把基准变成训练框架） |
| 核心贡献 | 提出 PAS/FAS + LLM-S3 基准 | 提出 SCSP 管道：数据重构 + 概率对齐 + 轻量微调 |
| 状态 | 完整（19 页） | **未完成**（9 页，Introduction 空、FAS 实验缺失、表格占位） |

**一句话关系**：基础工作证明"LLM 直接提示做社会调查模拟不够好，且 FAS 失败率高"；毕设用"真实调查分布驱动的数据设计 + 小模型微调"来补这个洞。

---

## 2. 基础工作：LLM-S3 基准

### 2.1 两个范式

**PAS（Partial Attribute Simulation，部分属性模拟）**

- 输入：受访者画像的一部分（人口统计 + 若干条件属性）
- 任务：预测缺失属性 $A_{target} \sim P_\theta(A_{target} \mid A_{prior})$
- 两种任务类型：
  - **数值预测**：目标是拟合总体分布，用 **KL-based score**（KL 散度的单调递减变换，越高越好）
  - **多选题预测**：逐人分类，用 **accuracy**
- 两种评测：zero-shot、few-shot（5 个示例）

**FAS（Full Attribute Simulation，全属性模拟）**

- 任务：直接生成完整的多属性合成数据集 $x \sim P_\theta(X \mid C, S)$ 或 $P_\theta(X \mid C, S, P(A_{prior}))$
- 两种条件：
  - **zero-context**：不给任何分布信息，靠模型参数知识
  - **context-enhanced**：注入真实边际分布（如性别、年龄、州分布）
- 评测：KL-based score
- 两类典型失败：
  - Case 1：输出样本数不足（要求 50 条只给 20 条）
  - Case 2：输出完全无关（写成散文而非结构化数据）

### 2.2 基准规模

11 个数据集、4 个领域：

| 领域 | 数据集 | 缩写 | 年份 | 样本量 | 属性数 |
|---|---|---|---|---|---|
| Social and Public Affairs | American National Election Studies | ANES | 2020 | 3,080 | 470 |
| | General Social Survey | GSS | 2022 | 4,150 | 1,156 |
| | Basic Income Survey | BIS | 2016–2017 | 9,649 | 15 |
| Work and Income | Employee Survey | EmpS | — | 3,025 | 23 |
| | American Community Survey | ACS | — | 2,000 | 13 |
| Household and Behavioral Patterns | Residential Energy Consumption Survey | RECS | 2020 | 18,496 | 799 |
| | National Household Travel Survey | NHTS | 2017 | 923,572 | 229 |
| | Trell Social Media Usage Data | Trell SMU | 2020 | 488,877 | 26 |
| Health and Lifestyle | Music & Mental Health Survey | MxMH | 2022 | 736 | 33 |
| | Young People Survey | YPS | 2013 | 1,011 | 150 |
| | Mental Health Dataset | MHD | 2014–2016 | 292,364 | 17 |

评测模型：GPT-3.5 Turbo、GPT-4 Turbo、LLaMA 3.0-8B、LLaMA 3.1-8B。

### 2.3 主要结论 `[论文声称]`

1. GPT 系列在选择题上更强（46.82% vs 43.04%），LLaMA 在数值预测上略好（0.5255 vs 0.5194）。
2. few-shot 对选择题有帮助（45.32% vs 44.54%），但对数值预测反而 zero-shot 更好（0.5296 vs 0.5154）。
3. context-enhanced 通常优于 zero-context，但存在数据集/模型依赖的例外。
4. **FAS 失败率**：LLaMA 系 15–58%，GPT 系接近 0。原因归为模型规模与对齐优化。
5. 推理蒸馏（DeepSeek-R1-Distill）对 8B 提升 +40.6%，对 70B 反而 −22.9%。
6. 上下文提示对 GSS 四个公共支出议题提升 6.67%–23.62%。

### 2.4 基础工作的局限（= 毕设切入点）

- 只做 prompting，没有训练；作者在 Limitations 明确写"未来可用 fine-tuning / RAG"。
- FAS 失败率高、分布对齐差。
- 覆盖广但不深：每个数据集只选少量属性。

---

## 3. 毕设工作：SCSP 管道

### 3.1 总体思路

**Social Cognition Simulation Pipeline (SCSP)**：

```
真实社会调查
   ├─ ANES ──► 宽表→长表 ──► 交互特征 ──► 采样 ──► LoRA 微调 ──► PAS 模型
   └─ GSS  ──► 历史分布 ──► 多模型外推 ──► IPF ──► 五类任务 ──► LoRA 微调 ──► FAS 模型
                                                    │
                                        ┌───────────┴───────────┐
                                  个体级一致性              群体级分布一致性
                                  （MAE/RMSE/ACC）        （JS/KL，边际+联合）
```

核心主张：**社会认知不是靠提示"演"出来的，而是靠概率对齐的数据设计"学"出来的。**

### 3.2 PAS 侧（ANES，个体级）

#### 3.2.1 从宽表到长表

- 原始宽表：一行一个选民，多列是不同候选人的温度计评分（0–100）。
- 转换为长表：一行一个 **voter–candidate pair** $(v_i, c_j, y_{ij}, t_i)$。
- 样本增益约 30×（文档中一处写 5.3× 是针对群体数据）。
- 同一套转换也用于 **9 个社会群体**的态度评分。

#### 3.2.2 候选人元数据解耦

不直接学 $P(y \mid \text{voter}, \text{"Biden"})$，而是学 $P(y \mid v, \mathcal{M}[c])$：

```text
M[c] = (party, ideology, age, position, is_incumbent, ...)
```

目的是泛化到未见过的新候选人（如 2028 年候选人），避免记住具体人名。

#### 3.2.3 核心交互特征（70+ 个）

| 特征 | 定义 | 理论依据 |
|---|---|---|
| `party_match` | 同党 = 1，独立 = −1，异党 = 0 | Campbell et al. (1960) |
| `ideology_distance_7` | $\lvert lcself_v - ideology_c \rvert \in [0,6]$ | Downs (1957) |
| `demographic_similarity` | 党派 0.25 + 意识形态 0.25 + 种族 0.20 + 性别 0.15 + 年龄 0.15 | 社会认同理论 |
| `cross_pressure` | CP1：同党但意识形态距离 ≥3；CP2：异党但距离 ≤1；CP3：人口相似但异党 | Lavine et al. (2012) |

完整特征见 `data/anes/processed/特征说明文档.md`，生成脚本是 `codes/ANES_PAS/feature_engineer/create_interaction_features.py`（1,129 行）。

#### 3.2.4 九种采样策略

| 策略 | 目标 | 样本量（默认配置） |
|---|---|---|
| balanced | 匹配支持度边际 | 1,000 |
| hard_cases | 中间态度/边界 | 500 |
| diverse | 特征组合多样性 | 500 |
| extreme | 两端尾部 | 200 |
| party_ideology | 党派 × 意识形态核心组合 | 600 |
| demographic_sim | 相似度分层 | 400 |
| incumbent | 现任效应 | 300 |
| cross_pressure | 冲突信号（最难） | 300 |
| candidate_specific | 候选人均衡 | 150 × N |

#### 3.2.5 时间切分

- 设计：训练 2016–2020，测试 2022–2024，严格不重叠。
- 实际：`train_dataset.jsonl`（2,491）/ `test_dataset.jsonl`（400）遵守；但后续的 `train_dataset_fixed.jsonl`（8,727）混入了 731 条 2022 样本，`enhanced_test_dataset.jsonl`（1,373）包含 2016–2022，**存在泄漏风险**（详见第 8 节）。

#### 3.2.6 LoRA 微调配置

- 基座：`meta-llama/Llama-3.1-8B-Instruct`
- 秩 `r=8`，`alpha=32`，`dropout=0.1`
- 目标模块：`q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj`
- 3 epochs，lr 1e-4（large 版用 3e-5），有效 batch 16（large 版 32），max_len 512
- 损失：只在答案 token 上计 causal LM loss

#### 3.2.7 群体态度扩展

- 9 个群体：Black / White / Hispanic / Asian Americans；Democratic / Republican Party；Socialists / Capitalists；Feminists。
- 长表 61,710 条，11,530 名受访者，2016–2022。
- 群体元数据 14 个字段（类别、意识形态、政治倾向、valence 等）。
- 训练/测试：早期 1,500/200；后续扩展为 `group_train_full.jsonl`（16,972）/ `group_test_balanced_v3.jsonl`（1,350）。

### 3.3 FAS 侧（GSS，群体级）

#### 3.3.1 多模型集成外推

对每个属性的每个取值、每个二阶交叉单元，用 4 个模型预测目标年：

```text
p̂(t*) = (1.0·p_poly + 1.0·p_loess + 0.6·p_exp + 0.8·p_linear) / 3.4
```

- 多项式回归、LOESS（tri-cube 核）、指数趋势、线性回归
- 结果裁剪到 [0,1] 并重新归一化
- 历史年份：2004, 2008, 2012, 2014, 2016, 2018, 2021, 2022（8 个）；目标年 2024

#### 3.3.2 IPF 联合分布

- 输入：3 个边际 + 3 个二阶联合
- 从独立近似 $Q^{(0)}(a,b,c)=P(A)P(B)P(C)$ 出发
- 交替乘性更新，迭代 800 次
- 归一化 + Laplace 平滑

#### 3.3.3 五类微调任务

| 任务 | 占比 | 内容 |
|---|---|---|
| A 批量生成 | 50% | 从联合分布采样 20 条，输出 JSON 数组 |
| B 先验采样 | 25% | 给定边际先验，生成匹配分布的样本 |
| C 情景变化 | 15% | 分布被扰动后的生成 |
| D 子群体分析 | 7% | 估计条件概率 + Wilson 95% 置信区间 |
| E 迭代修正 | 3% | 调整已生成样本以逼近目标分布 |

数据增强：70% 真实先验（±10% 扰动）+ 30% 随机先验；动态随机种子。

#### 3.3.4 GSS 属性配置

| 配置 | 三个属性 |
|---|---|
| EHC | `natenvir`（环境）、`natheal`（健康）、`natcity`（大城市） |
| EDC | `nateduc`（教育）、`natdrug`（毒品）、`natcrime`（犯罪） |
| 第三组 | `natfare`（福利）、`natsoc`（社保）、`nataid`（外援） |

每个属性取值 `1=too little, 2=about right, 3=too much`。

#### 3.3.5 生成与配额

- 三条生成路线：
  1. `DIFY_none_RAG.py`：无检索，直接调用 Dify API
  2. `DIFY_RAG.py`：带检索增强
  3. `FAS_GSS.py`：用 **outlines** 做结构化约束生成（Pydantic `BatchResponse`）
- 配额退火系统（`quota_annealing_system.py` + `DIFY_with_quota_annealing.py`）：给定配额（单属性/多属性、目标比例、容差、优先级），迭代生成并修正。
- 注意：v1 配额退火在实际论文数据中**未被采用**，v2 文档明确说"实际使用多模型集成外推，单次预测，无显式配额"。

### 3.4 评测体系

**PAS**

- MAE、RMSE：连续评分校准
- accuracy：
  - **relaxed accuracy**（当前代码实现）：预测与真值同属一个 20 分段，**或** 绝对误差 ≤ 10
  - **arg max choice accuracy**（论文描述的"选择题"）：在候选集合上取预测分最高者，与真实选择比较 —— **当前代码未实现**
- 分层：按候选人、策略、年份、群体

**FAS**

- 边际 JS / KL（对称）/ 卡方
- 联合 JS / KL（两两联合）
- TVD（部分脚本使用）
- 复现命令见第 9 节

### 3.5 论文状态（`毕设初稿_From_Prompting_to_Social_Cognition.pdf`，9 页）`[已验证]`

| 部分 | 状态 | 备注 |
|---|---|---|
| Title / Abstract | ✅ 完整 | 含具体数字（13%→60%、80% 降低） |
| 1 Introduction | ❌ 空 | 只有标题，没有正文 |
| 2 Related Works | ✅ 三段 | Social Representation / Competence vs Preference / Data-Centric |
| 3 Methodology | ✅ 完整 | 3.1 PAS、3.2 FAS、3.3 评测指标 |
| 4 Experiments | ⚠️ 部分 | 只有 4.1 PAS；FAS 实验缺失；正文写 "Table (candidates) and Table (groups)" 占位 |
| 5–7 页 | ❌ 模板残留 | 直接是 ACL 模板的 LaTeX 说明文字 |
| References | ✅ 约 20 条 | 需核对格式 |
| Appendix | ❌ 占位 | "Example Appendix / This is an appendix." |

---

## 4. 代码地图

| 模块 | 关键文件 | 职责 | 状态 |
|---|---|---|---|
| 特征工程 | `codes/ANES_PAS/feature_engineer/create_interaction_features.py` | 70+ 交互特征 | ✅ 可用 |
| | `feature_engineer/anes_wide_to_long.py` | 宽表→长表 | ✅ |
| | `feature_engineer/variable_extract.py` | 变量抽取 | ✅ |
| PAS 数据 | `codes/ANES_PAS/generate_finetune_data.py` | 9 种采样策略 + prompt 构造 | ✅ |
| | `codes/ANES_PAS/generate_temporal_split.py` | 时间切分 | ✅ |
| | `codes/ANES_PAS/generate_enhanced_test_set.py` | 增强测试集 | ✅ |
| | `codes/ANES_PAS/generate_dataset.py` | 早期单候选人生成 | ⚠️ 旧版 |
| PAS 训练 | `sft/finetune_anes_llama3.py` | 候选人 LoRA 微调 | ✅ |
| | `sft/finetune_anes_groups.py` | 群体 LoRA 微调 | ✅ |
| | `sft/finetune_anes_prefix.py` | Prefix-tuning 对照 | ✅ 脚本存在 |
| | `sft/finetune_hg.py` | GSS 通用微调 | ✅ |
| PAS 推理 | `sft/inference_anes.py` | 候选人推理 + 分层评测 | ✅ |
| | `sft/inference_anes_groups.py` | 群体推理 | ✅ |
| | `sft/inference_anes_prefix.py` | Prefix 推理 | ✅ |
| | `sft/compare_with_deepseek.py` / `compare_with_qwen3.py` / `compare_models.py` | 外部模型对比 | ✅ |
| FAS 数据 | `sft/teacher_distribution/get_distribution.py` | 历史分布统计 + 外推入口 | ✅ |
| | `get_distribution_dynamic.py` / `get_distribution_nonlinear*.py` | 4 模型集成 | ✅ |
| | `make_finetune_datasets.py` | IPF + 五类任务 + 数据生成 | ✅ |
| | `quota_annealing_system.py` | 配额退火 v1 | ⚠️ 实际未采用 |
| GSS 生成 | `codes/GSS/DIFY_none_RAG.py` | Dify 无检索生成 | ✅ |
| | `codes/GSS/DIFY_RAG.py` | Dify 检索增强 | ✅ |
| | `codes/GSS/FAS_GSS.py` | outlines 约束生成 | ✅ |
| | `codes/GSS/DIFY_with_quota_annealing.py` | 配额退火整合 | ⚠️ import 路径有问题 |
| GSS 评测 | `codes/GSS/evaluation.py` | JS/KL/卡方/TVD，边际+联合 | ✅ |
| 部署 | `sft/simple_merge_lora.py` | 合并 LoRA | ✅ |
| | `sft/convert_to_q4.py` | GGUF Q4 量化 | ✅ |
| | `sft/to_ollama.py` / `ollama_export/` | 导出 Ollama | ✅ |
| 无关目录 | `outlines/`、`sft/ollama_export/llama.cpp/`、`agent/booklearn/`、`demo.py` | 第三方/试验代码 | 🗑️ 可清理 |

---

## 5. 数据资产清单

### 5.1 ANES

| 路径 | 规模 | 内容 |
|---|---|---|
| `data/anes/raw_data/anes_pilot_2016.csv` | 2.7 MB | 2016 原始 |
| `data/anes/raw_data/anes_pilot_2018.csv` | 6.4 MB | 2018 原始 |
| `data/anes/raw_data/anes_pilot_2019.csv` | 9.4 MB | 2019 原始 |
| `data/anes/raw_data/anes_pilot_2020ets_csv.csv` | 3.7 MB | 2020 原始 |
| `data/anes/raw_data/anes_pilot_2022_csv_20221214.csv` | — | 2022 原始 |
| `data/anes/raw_data/anes_pilot_2024_20240319.csv` | — | 2024 原始 |
| `data/anes/raw_data/candidate_metadata.csv` | 29 人 | 候选人元数据 |
| `data/anes/processed/anes_all_years_with_features.csv` | 39 MB | 131,398 条 × 特征 |
| `data/anes/processed/X_features_classification.csv` | 41 MB | 分类特征矩阵 |
| `data/anes/finetune/train_dataset.jsonl` | 2,491 | 严格时间切分训练集（2016–2020） |
| `data/anes/finetune/test_dataset.jsonl` | 400 | 严格时间切分测试集（2022+2024） |
| `data/anes/finetune/train_dataset_fixed.jsonl` | 8,727 | 后续主训练集（含 731 条 2022） |
| `data/anes/finetune/train_dataset_large.jsonl` | 7,790 | large 版训练集（年份元数据缺失较多） |
| `data/anes/finetune/enhanced_test_dataset.jsonl` | 1,373 | 增强测试集（8 候选，2016–2022） |
| `data/anes/finetune/test_2024_dataset.jsonl` | 449 | 2024 专项测试集 |
| `data/anes/finetune/split_datasets/` | train 45 MB / val 15 MB / test 15 MB | 重新划分的全量数据集 |
| `data/anes/extend_features/data/anes_all_years_groups_long.csv` | 61,710 | 9 群体的 voter–group 长表 |
| `data/anes/extend_features/data/group_train_full.jsonl` | 16,972 | 群体训练集（后续） |
| `data/anes/extend_features/data/group_test_balanced_v3.jsonl` | 1,350 | 群体测试集（后续） |
| `data/anes/extend_features/data/group_train_enhanced.jsonl` | 1,500 | 群体训练集（早期） |
| `data/anes/extend_features/data/group_test_enhanced.jsonl` | 200 | 群体测试集（早期） |

### 5.2 GSS

| 路径 | 规模 | 内容 |
|---|---|---|
| `data/gss/GSS2004.csv` … `GSS2024.csv` | 3–8 MB / 年 | 9 个年份的原始调查 |
| `data/gss/GSS2024_sampled_ehc.csv` | 1,552 | 2024 真实分布（EHC 三题） |
| `data/gss/GSS2024_sampled_edc.csv` | 1,601 | 2024 真实分布（EDC 三题） |
| `data/gss/test_results/*.csv` | 每份 400 行左右 | 各模型/配置的生成结果 |
| `data/gss/test_results/*_report.json` | — | 配额退火报告 |

### 5.3 微调数据（GSS FAS）

| 路径 | 规模 | 内容 |
|---|---|---|
| `sft/JSON_data/balanced_finetune_dataset.json` | 470 | 五类任务均衡数据集（EHC） |
| `sft/JSON_data/combined_multiattr_dataset_3760.json` | 3,760 | 多属性合并数据集（test2/test3 使用） |
| `sft/JSON_data/multiattr_dataset_config_1_2000.json` | 2,000 | 配置 1 |
| `sft/JSON_data/multiattr_dataset_config_2_2000.json` | 2,000 | 配置 2 |

---

## 6. 实验记录（可验证结果）

### 6.1 PAS 候选人 `[已验证]`

测试集：`enhanced_test_dataset.jsonl`（1,373 条，8 位候选人）

| 模型 | MAE | RMSE | relaxed ACC |
|---|---|---|---|
| Base Llama-3.1-8B（few-shot） | 30.85 | 38.64 | 31.9% |
| Base Llama-3.1-8B（zero-shot） | 31.48 | 40.12 | 33.4% |
| **PAS 微调（v2, few-shot）** | **25.27** | 35.37 | **46.0%** |
| 严格切分测试集（400 条）：base | 31.53 | 35.84 | 16.0% |
| 严格切分测试集（400 条）：微调 | 26.38 | 36.58 | 41.8% |

### 6.2 PAS 群体态度 `[已验证]`

早期测试集（200 条，与训练同源）：

| 模型 | MAE | RMSE | ACC |
|---|---|---|---|
| Base | 27.01 | 32.45 | 27% |
| 微调（checkpoint-400） | 26.43 | 34.81 | 43% |

后续测试集（1,350 条，balanced_v3）：

| 模型 | MAE | RMSE | ACC |
|---|---|---|---|
| Base | 25.14 | 29.83 | 24.9% |
| 微调（同上） | **48.28** | 56.71 | 22.3% |

> ⚠️ 群体模型只在 0.377 epoch 时保存了 checkpoint-400（训练被中断），且在两个测试集上结论相反。**该模型不可信，需要重训**。

### 6.3 2024 时间偏移 `[已验证]`

测试集：`test_2024_dataset.jsonl`（449 条）

| 模型 | MAE | RMSE | ACC |
|---|---|---|---|
| Llama-3.1 base（few-shot） | **23.72** | 30.90 | **43.0%** |
| Llama-3.1 base（zero-shot） | 23.86 | 31.73 | 43.0% |
| Qwen3-4B（zero-shot） | **23.28** | 30.25 | **45.4%** |
| GPT-OSS-20B（few-shot） | 45.63 | 55.26 | 24.5% |
| **ANES 微调（zero-shot）** | **37.61** | 49.53 | 38.3% |

按策略（微调 vs base，MAE）：

| 策略 | 微调 | Base | 恶化 |
|---|---|---|---|
| cross_pressure_2024 | 55.54 | 29.64 | **+87%** |
| party_ideology_2024 | 40.14 | 24.38 | +65% |
| extreme_2024 | 30.67 | 18.70 | +64% |
| balanced_2024 | 34.71 | 23.13 | +50% |

根因诊断（详见 `data/anes/finetune/2024年性能问题诊断报告.md`）：

1. 时间分布偏移（训练 2016–2020，测试 2024；政治格局变化 + 极化加剧）。
2. 交叉压力样本占比低（约 3.8%），模型退化为"同党高分/异党低分"的简单规则。
3. 党派 × 意识形态组合分布偏移（2024 年复杂组合占比更高）。
4. 候选人特定效应（Biden 现任/年龄、Trump 诉讼/运动）。

### 6.4 外部模型对比 `[已验证]`

DeepSeek-R1-Distill-Llama-8B vs ANES 微调（严格切分 400 条）：

| 模型 | MAE | RMSE | ACC |
|---|---|---|---|
| DeepSeek-R1-Distill | 31.53 | 35.84 | 18.75% |
| ANES 微调 | 30.89 | 41.49 | 39.25% |

### 6.5 GSS FAS 分布指标 `[已验证，本次重算]`

> 用 `codes/GSS/evaluation.py` 重新计算（原脚本路径写死 `/home/cyyuan`，已临时改路径运行，未改动仓库）。
> `边际JS均值` = 三个属性 JS 的平均；`联合JS均值` = 三个两两联合 JS 的平均。

**EHC（natenvir / natheal / natcity）**

| 配置 | 生成样本 | 边际 JS 均值 | 联合 JS 均值 |
|---|---|---|---|
| `gss_31.csv`（baseline） | 351/400 | 0.0319 | 0.1001 |
| `gss_31_RAG.csv` | 379/400 | 0.0404 | — |
| `gss_31_sft.csv` | 400/400 | 0.0167 | 0.0470 |
| `gss_31_none_RAG_sft_ehc_v2.csv` | 399/400 | 0.0167 | 0.0484 |
| `gss_31_none_RAG_sft_ehc_v3.csv` | 400/400 | **0.0077** | **0.0363** |
| `gss_quota_ehc.csv`（配额退火） | 400/400 | 0.0175 | 0.0522 |

**EDC（nateduc / natdrug / natcrime）**

| 配置 | 生成样本 | 边际 JS 均值 | 联合 JS 均值 |
|---|---|---|---|
| `gss_31_edc.csv`（baseline） | 400/400 | 0.0552 | 0.1181 |
| `gss_31_RAG_edc.csv` | 376/400 | 0.0473 | 0.1096 |
| `gss_31_sft_edc.csv` | 381/400 | **0.0048** | **0.0160** |
| `gss_31_RAG_sft_edc.csv` | 381/400 | 0.0054 | 0.0176 |
| `gss_31_none_RAG_sft_edc_v2.csv` | 400/400 | 0.0050 | 0.0171 |
| `gss_31_none_RAG_sft_edc_v3.csv` | 400/400 | **0.0040** | 0.0215 |
| `gss_quota_edc.csv`（配额退火） | 395/400 | 0.0091 | 0.0216 |

**对论文口径的意义**：

- EDC 边际 JS 从 0.0552 → 0.0040（−93%），联合 JS 0.1181 → 0.0160（−86%），**支持"降低 80%+"的说法**。
- EHC 边际 JS 0.0319 → 0.0077（−76%），联合 0.1001 → 0.0363（−64%），**未达到 80%**。
- 配额退火版本没有优于普通 SFT 版本（EHC 甚至更差）。

### 6.6 SFT 模型清单

| 目录 | 用途 | 训练数据 | 轮数 | eval_loss | 备注 |
|---|---|---|---|---|---|
| `sft/sft_models/anes_llama3_lora` | ANES 候选人 | train_dataset（2,491） | 3 | 1.424 | 首版 |
| `sft/sft_models/anes_llama3_lora_large` | ANES 候选人 | train_dataset_large（7,790） | 3 | 1.161 | lr 3e-5, batch 32 |
| `sft/sft_models/anes_llama3_lora_v2` | ANES 候选人 | train_dataset_fixed（8,727） | 3 | 1.143 | 最新主模型 |
| `sft/sft_models/anes_groups_llama3_lora` | ANES 群体 | group_train_full | 0.377（中断） | 1.492 | **不可用** |
| `sft/sft_models/test1` | GSS FAS | 470 条均衡集 | 3 | — | loss 0.416 |
| `sft/sft_models/test2` | GSS FAS | 3,760 条合并集 | 10 | — | loss 0.416 |
| `sft/sft_models/test3` | GSS FAS | 3,760 条合并集 | 10 | — | loss 0.423，用于 gss3 |
| `sft/merged_model/gss2_q4` | GSS 量化模型 | — | — | — | Q4 GGUF |
| `sft/merged_model/gss3_q4` | GSS 量化模型 | — | — | — | Q4 GGUF |

---

## 7. 已完成清单 ✅

### A. 基础工作

- [x] A1 完成 PAS / FAS 两范式的形式化定义与提示模板设计
- [x] A2 构建 LLM-S3 基准：11 数据集 × 4 领域
- [x] A3 完成 4 个模型 × zero/few-shot × zero/context-enhanced 的评测
- [x] A4 完成 FAS 失败模式分析与失败率统计
- [x] A5 完成上下文提示、推理蒸馏、few-shot 偏置等分析
- [x] A6 论文成稿（19 页）

### B. PAS 数据侧

- [x] B1 ANES 2016–2024 六期原始数据整理（`data/anes/raw_data/`）
- [x] B2 候选人元数据表（29 人，`candidate_metadata.csv`）
- [x] B3 宽表→长表转换（候选人 + 群体两条线）
- [x] B4 70+ 交互特征工程（`create_interaction_features.py`，1,129 行）
- [x] B5 9 种采样策略生成器（`generate_finetune_data.py`，795 行）
- [x] B6 严格时间切分数据集（2,491 / 400）
- [x] B7 大规模训练集（8,727 / 7,790）
- [x] B8 增强测试集（1,373，8 候选）+ 2024 专项测试集（449）
- [x] B9 群体态度数据（61,710 条，9 群体）+ 两代训练/测试集
- [x] B10 数据说明文档（`数据集说明文档.md`、`特征说明文档.md`）

### C. PAS 训练与评测

- [x] C1 候选人 LoRA 微调脚本 + 4 组训练实验
- [x] C2 群体 LoRA 微调脚本 + 1 组训练（中断）
- [x] C3 Prefix-tuning 对照脚本
- [x] C4 推理/评测脚本（MAE/RMSE/relaxed ACC + 候选人/策略/年份分层）
- [x] C5 与 DeepSeek-R1-Distill、Qwen3-4B、GPT-OSS-20B 的对比实验
- [x] C6 2024 性能问题诊断报告（含根因与三种解决方案）

### D. FAS 侧

- [x] D1 GSS 2004–2024 九个年份数据整理
- [x] D2 四模型集成外推（poly / LOESS / exp / linear）
- [x] D3 IPF 联合分布构建（800 次迭代）
- [x] D4 五类微调任务生成器（A–E）
- [x] D5 GSS 微调数据集（470 / 2,000×2 / 3,760）
- [x] D6 GSS LoRA 微调（test1/2/3 三组）
- [x] D7 三条生成路线（Dify 无 RAG / Dify RAG / outlines 约束）
- [x] D8 三组属性配置（EHC / EDC / natrace-natarms-nataid）
- [x] D9 配额退火系统（实现 + 两组运行）
- [x] D10 分布评测脚本（边际 + 联合的 JS/KL/卡方）
- [x] D11 模型合并、Q4 量化、Ollama 导出

### E. 论文与文档

- [x] E1 ARR 论文初稿：Abstract / Related Works / Methodology / 部分 Experiments
- [x] E2 方法论文档（中英文各两版，`sft/teacher_distribution/docs/`）
- [x] E3 PAS 数据方法论文档（`codes/ANES_PAS/Dataset_Methodology_PAS_FAS.md`）
- [x] E4 多份中文说明（数据集、特征、微调、诊断）

---

## 8. TODO 清单

### P0 — 论文投稿必需

| # | 事项 | 具体动作 | 关联文件 | 验收标准 |
|---|---|---|---|---|
| 1 | 补 Introduction | 写清 gap：prompting 不够 → 需要数据驱动的机制对齐；列 3 条贡献 | ARR 论文 | 1 页完整正文 |
| 2 | 补 FAS 实验章节 | 用第 6.5 节数据写 4.2：设置、baseline、主表、消融 | 论文 + `data/gss/test_results/` | 有表格、有结论 |
| 3 | 补 PAS 主结果表 | 把 "Table (candidates) and Table (groups)" 换成真实表格 | 论文 + 第 6.1/6.2 节 | 表格数字可追溯 |
| 4 | 统一指标口径 | 实现 arg max choice accuracy；明确 relaxed ACC 与 choice ACC 的区别 | `sft/inference_anes.py` | 两种指标都能复现 |
| 5 | 核对论文数字 | 13%→60%、0%→72%、80% reduction 逐条重算，对不上就改论文 | 论文 + 结果 JSON | 每个数字有出处 |
| 6 | 清理模板残留 | 删除 5–7 页 ACL 模板正文；重写 Appendix | ARR 论文 | 无 "The first line of the file..." |
| 7 | 补 Limitations | 时间偏移、群体模型、GSS 只有三属性、单一数据集来源 | 论文 | 独立章节 |
| 8 | 引用核验 | 检查 References 完整性、格式、是否缺引用 | 论文 | 无占位引用 |

### P0 — 实验正确性

| # | 事项 | 具体动作 | 关联文件 | 验收标准 |
|---|---|---|---|---|
| 9 | 处理 2024 时间偏移 | 二选一：①把"迁移性"写成论文贡献（推荐）；②加入时序自适应/领域自适应实验。不能装作不存在 | `2024年性能问题诊断报告.md` | 论文有专门小节 |
| 10 | 清理数据泄漏 | 重新生成训练/测试：训练严格 ≤2020，测试 ≥2022；或明确声明"混合时间评测"并解释 | `train_dataset_fixed.jsonl`、`enhanced_test_dataset.jsonl` | 年份分布可审计 |
| 11 | 重训群体模型 | 当前 checkpoint 只有 0.377 epoch，且在 1,350 测试集上 MAE 48.3 vs base 25.1 | `sft/finetune_anes_groups.py` | 3 epochs + 两测试集一致 |
| 12 | 决定配额退火去留 | 数据显示配额版不优于 SFT；v2 文档也说未使用。建议从论文方法中删除或降为附录 | `quota_annealing_system.py` | 论文方法与代码一致 |

### P1 — 补实验

| # | 事项 | 具体动作 | 验收标准 |
|---|---|---|---|
| 13 | FAS 统一对比表 | 把 6.5 节整理成表：base / SFT / RAG / RAG+SFT / quota × EHC/EDC | 一张主表 + 一张消融表 |
| 14 | 跨领域扩展 | 目前只有 ANES + GSS；从基础工作 11 个数据集中选 1–2 个新领域（Work & Income 或 Health & Lifestyle）复现同一管道 | 新领域有数据 + 微调 + 结果 |
| 15 | 小模型矩阵补全 | Qwen3-4B、GPT-OSS-20B 目前只做推理，未微调；补 LoRA 微调对比 | 同表可比 |
| 16 | 消融实验 | 采样策略（9 种）、特征组（党派/意识形态/人口）、LoRA rank（4/8/16）、few-shot 数量 | 每组都有数字 |
| 17 | 多 seed 方差 | 当前结果多为单次运行；至少 3 seed，报告均值±标准差 | 表格带 std |

### P2 — 工程与可复现

| # | 事项 | 具体动作 | 影响 |
|---|---|---|---|
| 18 | 路径配置化 | 255 处 `/home/cyyuan` 硬编码，改为 `os.getenv("FAS_ROOT", ...)` 或 `pathlib` 相对路径 | 当前所有脚本直接跑都会 FileNotFoundError |
| 19 | 修复缺失模块 | `codes/GSS/hh2att.py` 引用 `prompt_gss.py`（不存在）；`run_quota.sh` 引用 `quota_annealing_runner.py`（不存在）；`DIFY_with_quota_annealing.py` 引用 `quota_annealing_system`（在 `sft/teacher_distribution/`，未在 sys.path） | 3 个脚本不可运行 |
| 20 | 环境统一 | `environment.yml` 只有基础依赖，缺 torch/transformers/peft/datasets/ollama/outlines/dify | 新机器无法一键复现 |
| 21 | 结果结构统一 | summary JSON 字段不统一（`overall` / `finetuned_model` / `evaluation` 混用） | 汇总脚本难写 |
| 22 | 清理无关目录 | `outlines/`（第三方库）、`sft/ollama_export/llama.cpp/`、`agent/booklearn/`、`demo.py` | 仓库体积与干扰 |
| 23 | 数据校验和 | 为关键 jsonl/csv 记录行数与 md5 | 防止版本混淆 |

### P3 — 后续扩展

| # | 事项 | 说明 |
|---|---|---|
| 24 | 更多社会学领域 | 基础工作 4 领域只扩展了 2 个 |
| 25 | 更多属性维度 | GSS 目前每组 3 属性，可扩到 4–5 维联合 |
| 26 | 更长时间序列 | 目前 8 个历史年份 → 预测 2024，可做滚动预测验证 |
| 27 | 部署 demo | Ollama 模型已有，可做交互式问卷模拟 demo |
| 28 | 伦理与偏差分析 | 群体态度的刻板印象风险、WEIRD 偏差 |

---

## 9. 复现指南

### 9.1 环境

```bash
conda activate FAS        # 或 /home/jpzhao_team/cyyuan/miniconda3/envs/FAS
```

已安装：pandas、numpy、scipy、pypdf、pdfplumber、pdfminer。
训练环境另需：torch、transformers、peft、datasets、accelerate。
生成环境另需：ollama、outlines、openai、requests、jinja2。

### 9.2 先修路径

所有脚本中的 `/home/cyyuan/FAS_PRO` 需要替换为 `/home/jpzhao_team/cyyuan/FAS_PRO`：

```bash
cd /home/jpzhao_team/cyyuan/FAS_PRO
grep -rl '/home/cyyuan' --include='*.py' --include='*.sh' sft codes data | \
  xargs sed -i 's#/home/cyyuan#/home/jpzhao_team/cyyuan#g'
```

> 建议先提交或备份，再批量替换。

### 9.3 关键命令（路径修正后）

**生成 PAS 数据**

```bash
cd codes/ANES_PAS
python generate_finetune_data.py --all
python generate_temporal_split.py
python generate_enhanced_test_set.py
```

**PAS 微调**

```bash
cd sft
python finetune_anes_llama3.py \
  --train_file ../data/anes/finetune/train_dataset_fixed.jsonl \
  --eval_file ../data/anes/finetune/enhanced_test_dataset.jsonl \
  --output_dir sft_models/anes_llama3_lora_v2 \
  --num_train_epochs 3 --learning_rate 1e-4 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 4
```

**PAS 评测**

```bash
python inference_anes.py \
  --lora_path sft_models/anes_llama3_lora_v2/final_model \
  --test_file ../data/anes/finetune/test_2024_dataset.jsonl \
  --temperature 0.3 --use_few_shot
```

**FAS 数据生成**

```bash
cd sft/teacher_distribution
python make_finetune_datasets.py --total_samples 2000
```

**GSS 生成**

```bash
cd codes/GSS
bash scripts/NoneRAG.sh
bash scripts/RAG.sh
```

**GSS 分布评测**

```bash
cd codes/GSS
python evaluation.py   # 需先把路径改为实际路径
```

---

## 10. 关键概念速查

| 概念 | 含义 |
|---|---|
| PAS | Partial Attribute Simulation，给定部分属性预测缺失属性 |
| FAS | Full Attribute Simulation，生成完整合成数据集 |
| SCSP | Social Cognition Simulation Pipeline，本文提出的训练管道 |
| LLM-S3 | Large Language Model-based Sociodemographic Survey Simulation，基础工作的基准名 |
| ANES | American National Election Studies，美国全国选举研究 |
| GSS | General Social Survey，美国综合社会调查 |
| 温度计评分 | 0–100 的"好感度"评分，PAS 的回归目标 |
| party_match | 选民与候选人党派是否一致（+1/0/−1） |
| ideology_distance | 7 点意识形态量表上的距离 |
| cross-pressure | 党派信号与意识形态信号冲突的情形 |
| IPF | Iterative Proportional Fitting，迭代比例拟合，用于重建联合分布 |
| 五类任务 | FAS 微调数据的 A–E 五种指令任务 |
| relaxed accuracy | 同 20 分段或误差 ≤10 即算对（当前代码口径） |
| choice accuracy | 在候选集合上 arg max 选人（论文口径，尚未实现） |
| JS / KL | Jensen–Shannon / Kullback–Leibler 散度，衡量分布差异 |

---

## 11. 风险与坑（务必先读）

1. **路径全部失效**：当前目录不是 `/home/cyyuan`，任何脚本直接跑都会失败。
2. **数据集年份不干净**：`train_dataset_fixed` 含 2022；`enhanced_test` 含 2016–2022。写论文前必须决定用哪套评测口径。
3. **指标不可比**：不同 summary 文件的 `accuracy` 都是 relaxed accuracy；论文里的"选择题准确率"是另一种指标，需要单独实现。
4. **群体模型不可用**：训练中断在 0.377 epoch，且两个测试集结论相反。
5. **配额退火未收敛**：EHC 0/4 满足、EDC 0/4 满足；且结果不优于普通 SFT，不建议作为论文主方法。
6. **论文数字待核**：Abstract 的 13%→60%、0%→72% 在本地结果中找不到对应记录，投稿前必须重算或改写。
7. **FS 侧只有 GSS**：跨领域扩展还没做，当前"不同社会学领域"的说法只覆盖 2/4 个领域。
8. **无版本管理习惯**：git 只有一个 initial commit，模型权重、数据、结果都在工作区，误删无法恢复。建议先做一次完整备份或提交。

---

## 12. 建议的推进顺序

如果只有有限时间，建议按这个顺序：

1. **先修路径 + 跑通一条最小链路**（PAS 推理 → GSS 评测），确认环境可用。
2. **重算论文所有数字**，列一张"论文声称 vs 本地实测"对照表。
3. **决定论文叙事**：是把 2024 退化写成"时间迁移性分析"（成本低、 scientically 诚实），还是补自适应实验（成本高、故事更完整）。
4. **补 FAS 实验章节**（数据已经齐了，只差写作和表格）。
5. **重训群体模型**（1 小时量级），否则群体部分不能写。
6. **再考虑跨领域扩展**（新数据集接入是最大工作量）。

---

*文档结束。若后续完成某项 TODO，请直接在对应条目打勾并注明日期。*
