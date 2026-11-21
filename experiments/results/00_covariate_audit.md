# Covariate availability audit

Fraction of rows whose value is blank, and the number of distinct political
profiles a model can actually see after the prompt builder substitutes defaults.

## Missing share by wave

| Wave | pid7 | ideo5 | lcself | educ | gender_resp | race_resp | birthyr_resp |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2016 | 1% | 0% | 0% | 0% | 0% | 0% | 0% |
| 2018 | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 2019 | 0% | 0% | 0% | 0% | 0% | 0% | 0% |
| 2020 | 0% | 100% | 0% | 0% | 100% | 0% | 0% |
| 2022 | 0% | 0% | 100% | 0% | 0% | 100% | 100% |
| 2024 | 100% | 100% | 100% | 100% | 0% | 0% | 0% |

## Prompt-visible political profiles

| Wave | rows | distinct profiles |
|---|---:|---:|
| 2016 | 10,727 | 69 |
| 2018 | 12,161 | 10 |
| 2019 | 21,347 | 75 |
| 2020 | 47,334 | 15 |
| 2022 | 3,169 | 71 |
| 2024 | 3,421 | 1 |

## pid7 direction and party-match sign

| Wave | corr(pid7, lcself) | corr(party_match, score) | pid7 out of range |
|---|---:|---:|---:|
| 2016 | +0.658 | +0.367 | 358 |
| 2018 | n/a | +0.052 | 12,161 |
| 2019 | +0.708 | +0.466 | 670 |
| 2020 | -0.546 | -0.223 | 48 |
| 2022 | n/a | +0.484 | 256 |
| 2024 | n/a | +nan | 0 |

## Split composition

| Split | rows by year |
|---|---|
| `train_r` | 2016: 629, 2018: 691, 2019: 1,229, 2020: 2,545 |
| `test_r` | 2016: 697, 2018: 727, 2019: 1,302, 2020: 2,868, 2022: 194, 2024: 226 |
| `train_t` | 2016: 806, 2018: 892, 2019: 1,580, 2020: 3,238 |
| `test_t` | 2024: 3,421 |
