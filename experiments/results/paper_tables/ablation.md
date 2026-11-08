# Ablations

| config | split | n | MAE | corr | choice | chance |
|---|---|---|---|---|---|---|
| Full (9-strategy) | in_dist | 2726 | 20.84 ± 0.24 | +0.679 ± 0.003 | 0.553 ± 0.011 | 0.163 |
| Full (9-strategy) | 2024 | 226 | 37.30 ± 1.41 | -0.056 ± 0.061 | 0.448 ± 0.018 | 0.500 |
| A1 random sampling | in_dist | 2726 | 21.27 | +0.658 | 0.543 | 0.163 |
| A1 random sampling | 2024 | 226 | 38.58 | +0.101 | 0.522 | 0.500 |
| A2 minimal prompt | in_dist | 2726 | 21.13 | +0.670 | 0.567 | 0.163 |
| A2 minimal prompt | 2024 | 226 | 38.48 | -0.136 | 0.451 | 0.500 |
| A3 name-only prompt | in_dist | 2726 | 21.45 | +0.657 | 0.565 | 0.163 |
| A3 name-only prompt | 2024 | 226 | 39.00 | -0.149 | 0.451 | 0.500 |