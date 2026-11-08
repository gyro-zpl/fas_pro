# Paired bootstrap (in-distribution 2016-2019)

Positive delta means the first model is worse (MAE) / better (corr, choice).

| comparison | n | ΔMAE [95% CI] | Δcorr [95% CI] | Δchoice [95% CI] |
|---|---|---|---|---|
| llm_seed42_minus_gbm | 2726 | +0.60 [-0.19, +1.42] | -0.039 [-0.056, -0.021] | +0.026 [-0.007, +0.062] |
| llm_seed43_minus_gbm | 2726 | +0.02 [-0.75, +0.76] | -0.035 [-0.051, -0.018] | +0.005 [-0.031, +0.038] |
| llm_seed44_minus_gbm | 2726 | +0.19 [-0.52, +0.93] | -0.032 [-0.048, -0.016] | +0.029 [-0.005, +0.065] |
| base_minus_gbm | 2726 | +3.58 [+2.95, +4.27] | -0.121 [-0.144, -0.100] | -0.055 [-0.103, -0.007] |
| llm_seed42_minus_base | 2726 | -2.98 [-3.82, -2.10] | +0.082 [+0.059, +0.107] | +0.081 [+0.038, +0.127] |