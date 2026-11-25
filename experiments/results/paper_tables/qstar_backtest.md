## Q* back-test at 2018

- waves used: [2004, 2008, 2012, 2014, 2016]
- real rows at 2018: 1105

| reconstruction | marginal JS | pairwise JS | three-way JS |
|---|---:|---:|---:|
| qstar | 0.0009 | 0.0037 | 0.0076 |
| persistence | 0.0011 | 0.0039 | 0.0078 |
| independence | 0.0009 | 0.0078 | 0.0194 |

Paired target bootstrap (2,000 resamples; negative = qstar closer):

| metric | qstar - persistence [95% CI] | P(qstar better) |
|---|---|---:|
| marginal | -0.00021 [-0.00042, +0.00000] | 0.974 |
| pair | -0.00023 [-0.00068, +0.00019] | 0.855 |
| three_way | -0.00019 [-0.00103, +0.00058] | 0.679 |

Internal consistency of Q* (L1 gaps):

- A marginal gap 0.000021
- B marginal gap 0.000320
- C marginal gap 0.000950
- B-C joint gap 0.000023

With the nonlinear projection, B/C marginal gaps can be nonzero because IPF only constrains the A marginal and the B-C joint; with the linear projection they are near zero because projection commutes with marginalization. Large A or B-C gaps indicate a projection bug.

## Q* back-test at 2022

- waves used: [2004, 2008, 2012, 2014, 2016, 2018, 2021]
- real rows at 2022: 1991

| reconstruction | marginal JS | pairwise JS | three-way JS |
|---|---:|---:|---:|
| qstar | 0.0016 | 0.0059 | 0.0119 |
| persistence | 0.0034 | 0.0093 | 0.0166 |
| independence | 0.0016 | 0.0092 | 0.0207 |

Paired target bootstrap (2,000 resamples; negative = qstar closer):

| metric | qstar - persistence [95% CI] | P(qstar better) |
|---|---|---:|
| marginal | -0.00181 [-0.00222, -0.00140] | 1.000 |
| pair | -0.00341 [-0.00417, -0.00267] | 1.000 |
| three_way | -0.00473 [-0.00577, -0.00371] | 1.000 |

Internal consistency of Q* (L1 gaps):

- A marginal gap 0.000022
- B marginal gap 0.000217
- C marginal gap 0.001350
- B-C joint gap 0.000023

With the nonlinear projection, B/C marginal gaps can be nonzero because IPF only constrains the A marginal and the B-C joint; with the linear projection they are near zero because projection commutes with marginalization. Large A or B-C gaps indicate a projection bug.

