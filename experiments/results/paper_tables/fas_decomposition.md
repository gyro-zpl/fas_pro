# FAS error decomposition

Distance to Q* uses the same JS definitions as the thesis (mean over the
three marginals and the three pairwise joints); the three-way column is the
full 27-cell divergence. The real-2024 columns are copied from `gss_fas.md`.

| config | run | n | to Q*: marginal | to Q*: pairwise | to Q*: 3-way | to real: marginal | to real: pairwise |
|---|---|---:|---:|---:|---:|---:|---:|
| EDC | base | 400 | 0.0503 | 0.1044 | 0.1612 | 0.0552 | 0.1181 |
| EDC | SFT | 381 | 0.0057 | 0.0142 | 0.0295 | 0.0048 | 0.0160 |
| EDC | SFT-v3 | 400 | 0.0046 | 0.0188 | 0.0457 | 0.0040 | 0.0215 |
| EDC | RAG | 376 | 0.0427 | 0.0935 | 0.1517 | 0.0473 | 0.1096 |
| EHC | SFT-v3 | 400 | 0.0065 | 0.0307 | 0.0771 | 0.0077 | 0.0363 |

The prior's own distance to the real 2024 sample is 0.0004 / 0.0076 (EDC)
and 0.0011 / 0.0225 (EHC). If a run's distance to Q* is larger than that,
most of its error is generation, not reconstruction.

