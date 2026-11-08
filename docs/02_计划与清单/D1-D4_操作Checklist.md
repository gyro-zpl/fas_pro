# 近期改动 Checklist（D1–D4）

> **对应计划**：`docs/02_计划与清单/20天冲刺计划.md`
> **目标**：用 4 天搭好"干净数据 + 正确指标 + 基线对照"，并通过 Gate G1
> **原则**：新增代码全部放在 `experiments/`，不重写旧脚本；旧结果只读，不覆盖

---

## 总览

| 天 | 主题 | 关键产出 | 预计工时 |
|---|---|---|---|
| D1 | 基建 | `experiments/` 骨架、路径配置、smoke test | 4–6 h |
| D2 | 数据 | 清洗后的候选人表 + 4 套 split + 审计报告 | 8–10 h |
| D3 | 指标 | 指标库、评分分布读出、评测 CLI、单测 | 8–10 h |
| D4 | 基线 | GBM/Ridge、base 模型、旧模型复测、G1 表 | 6–8 h |

---

# D1：基建与冒烟测试

## 1.1 备份（30 min）

```bash
cd /home/jpzhao_team/cyyuan
mkdir -p backup
rsync -av --exclude='sft/sft_models/*/checkpoint-*' \
          --exclude='sft/ollama_export/llama.cpp' \
          --exclude='outlines' \
          FAS_PRO/{codes,sft,data/anes/finetune,data/anes/extend_features,data/gss,docs} \
          backup/FAS_PRO_$(date +%Y%m%d)/
```

- [ ] 备份完成，记录大小
- [ ] 确认不包含模型权重（体积可控）

> 建议同时 `git init` 或把 `experiments/` 纳入版本管理，避免再次"一年后失忆"。

## 1.2 创建 experiments 骨架（30 min）

```bash
cd /home/jpzhao_team/cyyuan/FAS_PRO
mkdir -p experiments/{data,eval,baselines,tests,results}
touch experiments/__init__.py experiments/{data,eval,baselines,tests}/__init__.py
```

目标结构：

```
experiments/
├── config.py                 # 唯一路径来源
├── smoke_test.py
├── data/
│   ├── audit_raw.py
│   ├── clean_candidates.py
│   ├── build_splits.py
│   └── build_group_splits.py
├── eval/
│   ├── metrics.py
│   ├── readout.py
│   └── evaluate.py
├── baselines/
│   └── tabular.py
├── tests/
│   ├── test_metrics.py
│   └── test_readout.py
├── results/
├── runs.csv
└── README.md
```

## 1.3 写 `experiments/config.py`（1 h）

必须包含：

```python
import os
from pathlib import Path

ROOT = Path(os.environ.get("FAS_ROOT", "/home/jpzhao_team/cyyuan/FAS_PRO"))

# 数据
ANES_FEATURES = ROOT / "data/anes/processed/anes_all_years_with_features.csv"
ANES_GROUPS_LONG = ROOT / "data/anes/extend_features/data/anes_all_years_groups_long.csv"
GSS_DIR = ROOT / "data/gss"
GSS_RESULTS = GSS_DIR / "test_results"

# 新数据
V3 = ROOT / "data/anes/v3"
V3.mkdir(parents=True, exist_ok=True)

# 模型
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_V2 = ROOT / "sft/sft_models/anes_llama3_lora_v2/final_model"
LORA_GROUP = ROOT / "sft/sft_models/anes_groups_llama3_lora"

# 结果
RESULTS = ROOT / "experiments/results"
SEEDS = [42, 43, 44]
```

- [ ] 所有路径可由环境变量 `FAS_ROOT` 覆盖
- [ ] `python -c "from experiments import config; print(config.ROOT)"` 正常

## 1.4 修复实际要用的脚本路径（1–2 h）

**只改这 7 个**，不要全库 sed：

```
sft/finetune_anes_llama3.py
sft/inference_anes.py
sft/finetune_anes_groups.py
sft/inference_anes_groups.py
codes/ANES_PAS/generate_finetune_data.py
codes/ANES_PAS/generate_temporal_split.py
codes/ANES_PAS/generate_enhanced_test_set.py
```

```bash
cd /home/jpzhao_team/cyyuan/FAS_PRO
for f in \
  sft/finetune_anes_llama3.py sft/inference_anes.py \
  sft/finetune_anes_groups.py sft/inference_anes_groups.py \
  codes/ANES_PAS/generate_finetune_data.py \
  codes/ANES_PAS/generate_temporal_split.py \
  codes/ANES_PAS/generate_enhanced_test_set.py; do
  sed -i 's#/home/cyyuan#/home/jpzhao_team/cyyuan#g' "$f"
done
```

- [ ] 改完 `grep -n "/home/cyyuan"` 这 7 个文件应为空
- [ ] 每个文件能 `python -m py_compile <file>` 通过

## 1.5 GPU 盘点（30 min）

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader > experiments/results/gpu_inventory.txt
cat experiments/results/gpu_inventory.txt
```

- [ ] 记录可用卡号、显存、是否独占
- [ ] 在 `experiments/runs.csv` 里规划卡位（例如 0/1/2 训练，3 推理）

## 1.6 smoke test（1–2 h）

`experiments/smoke_test.py` 要做：

1. 读 `config.py`
2. 加载 tokenizer
3. 加载 base model + LoRA（`peft.PeftModel`）
4. 取 `train_dataset_fixed.jsonl` 前 3 条
5. 用 chat template 构造 prompt
6. `model.generate(max_new_tokens=8)`
7. 打印 prompt 前 200 字 + 输出
8. 断言：输出可解析为 0–100 的数字（或至少非空）

```bash
CUDA_VISIBLE_DEVICES=0 python -m experiments.smoke_test
```

- [ ] 能跑通，3 条预测都打印
- [ ] 记录单条推理耗时

## 1.7 建进度日志（10 min）

`docs/03_进度与执行/进度日志.md`，每天三行：

```
## 2026-11-05
- 做了什么：
- 得到什么：
- 卡在哪：
```

## D1 验收

- [ ] `experiments/config.py` 可用
- [ ] smoke test 退出码 0
- [ ] 7 个脚本路径已修
- [ ] GPU 清单与卡位计划已写
- [ ] 备份完成

---

# D2：数据清洗与切分

## 2.1 审计脚本 `experiments/data/audit_raw.py`

必须输出 `experiments/results/00_audit.md` + `00_audit.json`，包含：

1. 总行数、年份分布、候选人数、受访者数
2. **重复/异常候选人名**：`obama.`、`hrc.`、大小写、空格
3. **群体混入候选人表**：`black`、`white`、`dem`、`rep` 的行数
4. `support_score` 缺失率、越界值（<0 或 >100）
5. **面板结构**：每年受访者数、跨年出现次数分布、train/test 年受访者重叠
6. **重复 prompt**：同一 `(respondent_id, year, candidate)` 的重复行数
7. 每年候选人集合（清洗前后对比）

已知基线（用于核对）：

| 指标 | 当前值 |
|---|---|
| 总行数 | 132,948 |
| 受访者 | 3,165 |
| 跨年受访者 | 3,080 |
| 尾点重复名行数 | 8,165 |
| 群体混入行数 | 23,060 |
| 2022/2024 受访者 | 1,909，全部在训练年出现过 |

- [ ] 审计报告生成
- [ ] 数字与上表一致（不一致要解释）

## 2.2 清洗脚本 `experiments/data/clean_candidates.py`

规则：

1. 候选人名：`strip()` → `lower()` → 去掉尾部 `.`
2. 过滤群体目标：`black`, `white`, `dem`, `rep` 从候选人表移除
3. 去重键：`(respondent_id, year, candidate)`，保留第一条，记录删除数
4. `support_score` 限制在 `[0, 100]`，缺失/越界删除
5. 输出：
   - `data/anes/v3/anes_candidates_clean.csv`
   - `data/anes/v3/clean_manifest.json`（清洗前后行数、删除原因计数）

- [ ] 清洗后 2024 候选人 = `biden`, `trump`
- [ ] 清洗后 2022 候选人 = `biden`, `trump`
- [ ] 清洗后无尾点名
- [ ] 清洗后无 `black/white/dem/rep`

## 2.3 切分脚本 `experiments/data/build_splits.py`

### 输入
`data/anes/v3/anes_candidates_clean.csv`

### 输出
```
data/anes/v3/splits/
├── train_t.jsonl        # 2016-2020
├── val_t.jsonl          # 2022
├── test_t.jsonl         # 2024
├── train_r.jsonl        # 80% respondents, 2016-2022
├── test_r.jsonl         # 20% respondents, 2016-2024
├── test_tr.jsonl        # 2024 ∩ test_r respondents
└── manifest.json
```

### 关键规则

1. **受访者哈希切分**（确定性）：
   ```python
   def is_test_respondent(rid: str) -> bool:
       return int(hashlib.md5(str(rid).encode()).hexdigest(), 16) % 10 < 2
   ```
2. **prompt 复用**：直接调用 `codes/ANES_PAS/generate_finetune_data.py` 中的
   `ANESFinetuneDataGenerator._create_prompt_from_row`，不要另写模板。
   - 若该函数依赖实例状态，抽出为模块级函数再调用。
3. **字段**：`instruction`, `input=""`, `output=str(int/float score)`,
   `metadata={respondent_id, year, candidate, strategy, true_score, party_match,
   ideology_distance_7, row_id}`
4. **manifest 必须记录**：
   - 每个文件的样本数、md5
   - 年份分布、候选人分布、受访者数
   - `train_t` vs `test_t` 的受访者重叠（预期 >0，需记录）
   - `train_r` vs `test_r` 的受访者重叠（预期 =0）
   - 跨 split 的重复 instruction 数（预期 =0）

### 验收断言（写在脚本里，失败即退出）

```python
assert set(train_t.year) & set(test_t.year) == set()
assert respondent_overlap(train_r, test_r) == 0
assert duplicate_instructions(all_splits) == 0
assert set(test_t.candidate) == {"biden", "trump"}
```

- [ ] 7 个文件生成
- [ ] manifest 通过全部断言
- [ ] 人工抽查 3 条 train / 3 条 test，prompt 可读

## 2.4 群体切分 `experiments/data/build_group_splits.py`

- 输入 `anes_all_years_groups_long.csv`
- 时间切分：train 2016–2020，test 2022
- 受访者切分：同样 md5 80/20
- 输出 `data/anes/v3/groups/{train_t,test_t,train_r,test_r}.jsonl` + manifest

- [ ] 4 个文件生成
- [ ] 群体目标 9 个齐全
- [ ] 受访者重叠断言通过

## D2 验收

- [ ] 审计报告解释清楚每个异常
- [ ] 清洗后候选人集合正确
- [ ] 4 套 split 可复现（同输入两次运行 md5 一致）
- [ ] manifest 断言全部通过
- [ ] 记录清洗前后可用样本量变化

---

# D3：指标与评分读出

## 3.1 `experiments/eval/metrics.py`

必须实现的函数（含类型标注和 docstring）：

```python
def mae(y_true, y_pred) -> float
def rmse(y_true, y_pred) -> float
def pearson(y_true, y_pred) -> float
def spearman(y_true, y_pred) -> float
def relaxed_accuracy(y_true, y_pred, bin_size=20, tol=10) -> float
def choice_accuracy(df, pred_col="pred", true_col="true_score",
                    group_cols=("respondent_id", "year"),
                    min_candidates=2) -> dict
def pairwise_accuracy(df, pred_col="pred", true_col="true_score",
                      group_cols=("respondent_id", "year")) -> dict
def ece(y_true, y_pred, n_bins=10) -> float
def bootstrap_ci(y_true, y_pred, metric_fn, n_boot=1000, seed=42) -> tuple
```

### choice_accuracy 返回

```python
{
  "accuracy": float,
  "n_groups": int,
  "chance": float,          # 1 / 平均候选人数
  "n_candidates_mean": float,
  "by_year": {...}
}
```

### 实现要点

- 每个 group 至少 `min_candidates` 个候选人，否则跳过
- 真实选择 = `true_score` argmax（rating-induced choice）
- 平票处理：预测平票时算错（并在报告里说明）
- 2024 只有 2 人 → chance = 0.5

- [ ] 单元测试覆盖：完美预测、随机预测、平票、单候选人组

## 3.2 `experiments/eval/readout.py`

### 第一步：tokenizer 检查

```python
def inspect_rating_tokens(tokenizer) -> dict:
    """返回 {n: token_ids} 以及 multi_token 列表"""
```

```bash
python -m experiments.eval.readout --inspect
```

预期输出：0–100 中哪些数字是多 token。
- 若全部单 token → 用 `first_token_logits` 方案
- 若存在多 token → 降级为粗网格（0, 5, …, 100）或 `outlines` 约束解码

### 第二步：读出类

```python
class RatingReadout:
    def __init__(self, tokenizer, mode="expectation"): ...
    def from_logits(self, logits) -> dict:
        """返回 {expectation, argmax, entropy, top5}"""
    def from_text(self, text) -> int | None:
        """fallback：正则解析"""
```

### 第三步：模型推理

```python
def predict_ratings(model, tokenizer, prompts, batch_size=8) -> list[dict]:
    """对每个 prompt 做一次 forward，取第一个生成位置的 logits"""
```

要点：
- 用 `apply_chat_template(..., add_generation_prompt=True, return_tensors="pt")`
- 只做 forward，不 `generate`，速度更快
- 只对 rating token 做 softmax
- 返回 `expectation`（用于 MAE）与 `argmax`（用于 choice）

- [ ] tokenizer 检查报告写入 `experiments/results/00b_tokenizer.md`
- [ ] `predict_ratings` 在 20 条样本上跑通
- [ ] 输出全部在 `[0, 100]`

## 3.3 `experiments/eval/evaluate.py`

CLI：

```bash
python -m experiments.eval.evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --lora /path/to/final_model \
  --split data/anes/v3/splits/test_t.jsonl \
  --prompt-mode zero \
  --readout expectation \
  --batch-size 8 \
  --output experiments/results/preds/v2_test_t_zero.jsonl
```

输出：
- `*.predictions.jsonl`：每条含 `instruction`, `true_score`, `pred_expectation`, `pred_argmax`, `entropy`, `metadata`
- `*.summary.json`：总体 + 按候选/年份/策略/群体的分层指标

- [ ] 支持 `--limit` 参数（先跑 50 条调试）
- [ ] 支持 `--prompt-mode {zero,few}`
- [ ] summary 里包含 choice accuracy

## 3.4 单元测试

```bash
python -m pytest experiments/tests -q
```

`test_metrics.py`：
- hand-computed MAE/RMSE
- relaxed accuracy 边界（19 vs 21）
- choice accuracy 的平票/单候选

`test_readout.py`：
- 构造假 logits，验证 expectation/argmax
- 多 token 数字的 fallback

- [ ] 测试全绿
- [ ] 覆盖率不要求，但核心分支必须覆盖

## D3 验收

- [ ] `pytest` 通过
- [ ] `evaluate.py --limit 50` 跑通
- [ ] 50 条样本的 MAE、choice accuracy 合理
- [ ] tokenizer 多 token 情况已记录并决定方案

---

# D4：基线与 Gate G1

## 4.1 表格基线 `experiments/baselines/tabular.py`

### 数据连接方式（已验证）

`metadata.row_id` = `anes_all_years_with_features.csv` 的 **0-based 行号**
（抽查 500 条，500 条匹配）。用法：

```python
df = pd.read_csv(config.ANES_FEATURES, low_memory=False)
row = df.iloc[meta["row_id"]]
```

> 注意：CSV 有 132,948 行；`row_id` 最大值需 < 行数。若清洗后重排 CSV，
> 必须在清洗脚本里保留原始 `row_id` 列，否则映射失效。

### 特征

- 数值列：年龄、教育、收入、pid7、ideo5、lcself、party_match、
  ideology_distance_7、demographic_similarity 等
- 类别列：race、gender、candidate、year、region 等 → one-hot
- 排除：`support_score`、`is_missing`、`respondent_id`

### 模型

```python
HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, random_state=seed)
Ridge(alpha=1.0)
```

### 评测

- 在 `test_t`、`test_r`、`test_tr` 上计算 MAE/RMSE/relaxed ACC
- choice accuracy 用同一套 group 逻辑

- [ ] 两个模型各 3 seeds
- [ ] 输出 `experiments/results/01_tabular.json`

## 4.2 base 模型评测

```bash
python -m experiments.eval.evaluate \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --split data/anes/v3/splits/test_t.jsonl \
  --prompt-mode zero --readout expectation \
  --output experiments/results/preds/base_test_t_zero.jsonl
```

同样跑 `few`、`test_r`、`test_tr`。

- [ ] 6 个预测文件（2 prompt × 3 split）
- [ ] summary 齐全

## 4.3 旧微调模型复测（量化泄漏）

用 `LORA_V2` 在上述 split 上跑同样 6 组。

**预期**：
- 在 `test_t`（2024）上，清洗后可能仍然差于 base；
- 在 `test_r`（含训练年份的陌生人）上应明显好于 base；
- 如果 `test_t` 与 `test_r` 差距很大，说明时间偏移确实是主因。

- [ ] 6 个预测文件
- [ ] 与旧 `enhanced_test` 结果对比，记录差异

## 4.4 生成 G1 表

`experiments/results/01_baselines.md`：

| Model | Test T MAE | RMSE | relaxed ACC | choice ACC | Test R MAE | choice ACC | Test T×R MAE | choice ACC |
|---|---|---|---|---|---|---|---|---|
| Ridge | | | | | | | | |
| HistGB | | | | | | | | |
| Base Llama zero | | | | | | | | |
| Base Llama few | | | | | | | | |
| LoRA v2 zero | | | | | | | | |
| LoRA v2 few | | | | | | | | |

- [ ] 表格填满
- [ ] 每个数字可追溯到 predictions 文件

## 4.5 Gate G1 判定

写 `experiments/results/G1_decision.md`，回答三个问题：

1. **清洗后主结果是否还成立？**
   - 若 LoRA 在 `test_r` 上仍显著优于 base → 成立
   - 若不成立 → 论文主张需改为"分布级对齐"而非"个体级提升"
2. **GBM 是否碾压 LLM？**
   - 若 GBM 明显更好 → LLM 的贡献不在 MAE，需转向选择/分布/泛化
   - 若接近 → 说明数据设计是主因，LLM 提供的是泛化与分布能力
3. **时间偏移是否仍是主要失败模式？**
   - 对比 `test_t` 与 `test_r` 的差距

**G1 通过条件**：三个问题都有明确答案，且团队接受据此调整后的论文主张。

- [ ] G1 决策文档完成
- [ ] 若需调整主张，更新 `docs/02_计划与清单/20天冲刺计划.md` 第 1.3 节

---

# 不要做的事（D1–D4）

- ❌ 不要重写 `finetune_anes_llama3.py` 的训练循环
- ❌ 不要全库 sed 替换路径（只改 7 个在用脚本）
- ❌ 不要在 G1 之前启动大规模训练
- ❌ 不要删除旧数据/旧结果（只读）
- ❌ 不要把 relaxed ACC 叫做 "accuracy"（论文里必须区分）
- ❌ 不要用 `enhanced_test_dataset.jsonl` 作为主评测集

---

# 完成后的下一步（D5–D7 预告）

1. 用 3 seeds 在 `train_t` 上训练 LoRA（主结果）
2. 用 3 seeds 在 `train_r` 上训练（受访者泛化）
3. 群体模型重训（3 epochs，不再是 0.377）
4. 四个消融各 1–2 seeds
5. 全部在 `test_t` / `test_r` / `test_tr` 上评测

预计 GPU 占用：4 卡 × 2 天可全部跑完。
