# Covariate-dropout probe

Year 2019, `test_r.jsonl`, 1,302 rows. The ablated prompt uses the 2024 defaults: Independent, Moderate, High school, Independent voter, ideology distance 0.0.

- distinct prompt profiles: original **423**, ablated **1**

| Condition | Model | n | MAE | Pearson | choice | chance | pred std |
|---|---|---:|---:|---:|---:|---:|---:|
| original | LLM | 1,302 | 19.339 | 0.714 | 0.665 | 0.150 | 38.93 |
| ablated | LLM | 1,302 | 31.667 | 0.166 | 0.356 | 0.150 | 12.18 |
| original | TABULAR | 1,302 | 19.038 | 0.734 | 0.649 | 0.150 | 25.00 |
| ablated | TABULAR | 1,302 | 31.460 | 0.172 | 0.356 | 0.150 | 6.99 |
