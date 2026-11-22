# paper_tables — 结果表索引

> 这个目录里的表由不同脚本、在不同协议下生成。**先看这张表，再引用数字。**
> 数字口径以 [`docs/口径总表_2026-11-21.md`](../../../docs/口径总表_2026-11-21.md) 为准。

## 现行（可引用）

| 文件 | 内容 | 生成脚本 | 协议 |
|---|---|---|---|
| `r_protocol.md/.json` | **PAS 主表** + 配对/聚类 bootstrap | `experiments/eval/r_protocol_tables.py` | R（波内编号留出） |
| `r_pas_by_year.md` | R 模型按年 | 同上 | R |
| `r_test_t.md` / `r_test_tr.md` | R 模型在 2024 / strict T×R | 同上 | R |
| `r_cross_pressure.md` | 交叉压力子群（R 模型） | 同上 | R |
| `gss_fas.md` | FAS 全部条件（JS/KL） | `gss_tables.py` | FAS |
| `fas_bootstrap.md` | FAS 行级 bootstrap CI | `fas_bootstrap.py` | FAS |
| `fas_coverage.md` | coverage-adjusted FAS | `fas_coverage.py` | FAS |
| `prior_baseline.md/.json` | prior-only 基线、CI、10 个采样种子范围、与 LLM 条件的独立 bootstrap 差 | `prior_baseline.py` | FAS |
| `pas_by_year.md` | T 协议按年（时间分析用） | `paper_tables.py` | T |
| `ablation.md` | 消融 A1–A3 | `ablation_table.py` | T，1 seed |

## legacy/（勿引用）

`legacy/` 存放 T 协议时期的旧表。对应的三个脚本（`bootstrap_main.py`、`paired_bootstrap.py`、
`cross_pressure.py`）现在也只写入这个目录，不会污染顶层。

| 文件 | 为什么过时 |
|---|---|
| `legacy/bootstrap_main.md/.json` | 用 T 协议模型 + 池化 choice（0.518）算的旧主表；已被 `r_protocol.md` 取代 |
| `legacy/second_family.md` | 旧 T 协议下的第二 checkpoint 对比（无生成脚本，手工表） |
| `legacy/cross_pressure.md` | 旧 T 协议交叉压力；R 版见 `r_cross_pressure.md` |
| `legacy/paired_bootstrap.md` | 旧 T 协议配对检验 |

## Prior-only 基线（已落盘）

`prior_baseline.md/.json` 由 `experiments/eval/prior_baseline.py` 生成，包含：

- 从重建的 $Q^*$ 直接抽 400 行的点估计与 1,000 次行级 bootstrap CI；
- 10 个采样种子（42–51）的范围；
- 与 v3 / SFT / base 的独立 bootstrap 差值 CI。

重建 $Q^*$ 需要 GSS 2004–2022 原始 CSV，仓库未包含；运行时用 `--data-dir`
指向这些文件所在目录。完整命令见 `prior_baseline.md` 的 Reproduce 段。

上一版正文里的种子范围（EDC 0.0004–0.0012 / 0.0064–0.0076，EHC
0.0011–0.0028 / 0.0190–0.0225）没有对应的已提交脚本，现已替换为当前脚本和
种子 42–51 的结果：EDC 0.0003–0.0009 / 0.0034–0.0076，EHC 0.0009–0.0033 /
0.0150–0.0239。点估计与旧值一致，所有抽样仍远低于 v3。
