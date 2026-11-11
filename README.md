# FAS_PRO

**From Prompting to Structured Alignment: Individual, Population, and Temporal Boundaries of LLM Survey Simulation**

本仓库是毕设工作的代码、数据切分、实验结果与论文初稿。核心工作是把 LLM 社会调查模拟从"prompting-only 评测"重构为**数据驱动的对齐 pipeline**，并在三个层面做系统评估：

- **PAS（个体级）**：给定选民画像与候选人，预测 0–100 的支持度评分，并由评分诱导选择。
- **FAS（群体级）**：从 GSS 历史分布重建联合分布，生成合成人口，评估边际/联合分布对齐。
- **时间边界**：2019→2024 的时间偏移下，LLM 与 GBM 的个体信号如何衰减。

---

## 主要结论

| 层面 | 结果 |
|---|---|
| PAS 分布内（2016–2019，陌生人） | 微调 Qwen3-8B：MAE **20.84**、corr **0.679**、choice **0.518**（chance 0.163）；HistGB：20.58 / 0.714 / 0.534；配对 bootstrap 显示 MAE/choice 差异不显著，GBM corr 更高 |
| PAS 微调必要性 | base Qwen：MAE 24.16、corr 0.593；微调相对 base 的 ΔMAE **−2.98**、Δcorr **+0.082**、Δchoice **+0.081** |
| FAS（GSS） | EDC 边际/联合 JS 降低 **91% / 86%**（全覆盖 v3：93% / 82%）；EHC：76% / 64% |
| 时间边界 | LLM corr：2019 **0.722** → 2020 0.478 → 2022 0.408 → 2024 **−0.056**；GBM 同样塌缩（−0.008）；recency + 2022 不缓解 |
| 数据审计 | 132,948 → **98,159** 行；尾点重复名 8,165、群体混入 23,060、异常评分 3,588 |

**一句话**：结构化对齐是必要的，但不是充分的——它能让 8B 模型在同分布陌生人上与强 GBM 持平，并在 EDC 上把分布差异降低约 90%，但无法阻止四年后的时间塌缩。

---

## 仓库结构

```text
.
├── codes/                 # 早期数据/生成脚本（ANES PAS、GSS FAS）
├── experiments/           # 本次重构的独立实验层
│   ├── data/              # 审计、清洗、T/R/T×R 切分、消融数据
│   ├── eval/              # 指标、读出、评测 CLI、bootstrap、论文表/图
│   ├── baselines/         # HistGB / Ridge
│   ├── train_lora.py      # Qwen/Llama 通用 LoRA 训练
│   ├── diagnostics/       # prompt / readout 诊断
│   └── results/           # paper_tables、轻量结果、runs.csv
├── sft/                   # 早期 LoRA/推理/量化脚本
├── data/
│   ├── anes/v3/           # 清洗后的候选人表 + 切分（30M + 16M）
│   ├── gss/               # 2024 真实分布样本 + 生成结果（小文件）
│   └── anes/processed/    # 特征说明与数据集说明文档
├── docs/
│   ├── 01_背景与总览/     # 项目总览、基础工作论文
│   ├── 02_计划与清单/     # 20 天计划、操作 checklist
│   ├── 03_进度与执行/     # 时间线、执行记录、运行快照、最终总结
│   └── 04_论文稿件/       # 完整初稿、分章 md、图
├── doc_latex/             # ACL LaTeX 版本（main.tex / main.pdf / references.bib）
├── thesis/                # UMAC 英文毕业论文（main.tex / main.pdf，正文 10 页）
└── environment.yml
```

---

## 数据与模型说明

- **原始调查数据未包含**：ANES（2016–2024）与 GSS（2004–2024）为公开数据，请从官方渠道下载后放到 `data/anes/raw_data/` 与 `data/gss/`。
- **模型权重未包含**：LoRA adapter 与合并/量化模型体积过大（13GB），未推送到仓库；可用 `experiments/train_lora.py` 重新训练。
- **包含的派生数据**：清洗后的候选人表（`data/anes/v3/anes_candidates_clean.csv`）与可复现切分（`data/anes/v3/splits/`），便于直接复现评测。
- **包含的生成结果**：`data/gss/test_results/` 下的 EDC/EHC 生成 CSV（小文件），可直接复现 FAS 表。
- **Llama 未评测**：Llama-3.1-8B 为 gated 模型，本工作未能获得访问权限；论文 Limitations 已说明单模型家族限制。

---

## 复现

环境：

```bash
conda activate FAS   # 或使用 environment.yml
# 本次实验使用 llama_factory 环境（torch + transformers + peft + datasets + sklearn）
```

数据管线：

```bash
python -m experiments.data.audit_raw
python -m experiments.data.clean_candidates
python -m experiments.data.build_splits
python -m experiments.data.build_group_splits
python -m experiments.data.build_ablation_splits
```

训练与评测：

```bash
# 主模型（3 seeds）
bash experiments/run_train_seeds.sh

# 评测
bash experiments/run_eval_suite.sh 3 42:test_t:generate 43:test_t:generate 44:test_t:generate

# 消融
bash experiments/run_ablations_suite.sh 1
```

论文表与图：

```bash
python -m experiments.eval.paper_tables
python -m experiments.eval.ablation_table
python -m experiments.eval.bootstrap_main
python -m experiments.eval.paired_bootstrap
python -m experiments.eval.gss_tables
python -m experiments.eval.fas_coverage
python -m experiments.eval.cross_pressure
python -m experiments.eval.make_figures
```

---

## 论文初稿

- 完整初稿：`docs/04_论文稿件/初稿_2026-11-09.md`（约 7,700 词，§1–§10 + Abstract）
- 分章源文件：`docs/04_论文稿件/chapters/`
- 图：`docs/04_论文稿件/figures/`
- 结果登记：`docs/03_进度与执行/论文可用结果_2026-11-07.md`
- 最终总结：`docs/03_进度与执行/项目最终总结_2026-11-09.md`

### LaTeX 版本

- `doc_latex/main.tex`：ACL 模板（`[review]` 模式）
- `doc_latex/main.pdf`：已编译版本（15 页）
- `doc_latex/references.bib`：参考文献
- `doc_latex/tables/`、`doc_latex/figures/`：表格与图
- 编译：`cd doc_latex && tectonic main.tex`（或 `pdflatex` + `bibtex`）
- 说明：`body_final.tex` 由 `build_md.py` + `assemble.py` 从 `docs/04_论文稿件/chapters/` 生成；改正文请改 chapters 后重跑脚本。

### 英文毕业论文（UMAC 模板）

- `thesis/main.tex`：UMAC FST 硕士论文模板（A4、12pt Times、1.5 倍行距）
- `thesis/main.pdf`：已编译版本（**正文 10 页**，总 26 页）
- `thesis/chapters/`：5 章正文；`thesis/appendix/`：4 个附录（含第二模型家族 DeepSeek-R1-Distill-Llama-8B）
- 编译：`cd thesis && tectonic main.tex`
- 作者：CHENYU YUAN；导师：PENGYANG WANG；日期：April 2027

---

## 引用

```bibtex
@misc{fas_pro_2026,
  title  = {From Prompting to Structured Alignment: Individual, Population, and Temporal Boundaries of LLM Survey Simulation},
  author = {Yuan, Chenyu},
  year   = {2026},
  note   = {Undergraduate thesis project}
}
```
