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
| `pas_by_year.md` | T 协议按年（时间分析用） | `paper_tables.py` | T |
| `ablation.md` | 消融 A1–A3 | `ablation_table.py` | T，1 seed |

## legacy（勿引用）

| 文件 | 为什么过时 |
|---|---|
| `bootstrap_main.md/.json` | 用 T 协议模型 + 池化 choice（0.518）算的旧主表；已被 `r_protocol.md` 取代 |
| `second_family.md` | 旧 T 协议下的第二 checkpoint 对比 |
| `cross_pressure.md` | 旧 T 协议交叉压力；R 版见 `r_cross_pressure.md` |
| `paired_bootstrap.md` | 旧 T 协议配对检验 |

## 未落盘的数字

`prior-only` 基线（EDC 0.0004/0.0076、EHC 0.0011/0.0225）由
`docs/03_进度与执行/prior_baseline_check.py` 在服务器上生成，需要 GSS 2004–2022
原始 CSV（仓库未包含），目前没有对应的 results 文件。引用时注明来源与运行命令。
