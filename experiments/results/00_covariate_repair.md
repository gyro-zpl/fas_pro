# Covariate repair audit

Reconstructed from the raw ANES CSVs without modifying `data/anes/v3`.

| Wave | Rows | pid7 before/after | ideo5 before/after | educ before/after | lcself before/after | Profiles before/after | corr(pid7, lcself) |
|---|---:|---|---|---|---|---|---:|
| 2016 | 10,727 | 99% / 95% | 100% / 93% | 100% / 100% | 100% / 100% | 1156 / 1075 | +0.658 |
| 2018 | 12,161 | 100% / 96% | 100% / 89% | 100% / 100% | 100% / 100% | 287 / 1021 | +0.727 |
| 2019 | 21,347 | 100% / 97% | 100% / 93% | 100% / 100% | 100% / 100% | 1330 / 1209 | +0.708 |
| 2020 | 47,334 | 100% / 100% | 0% / 100% | 100% / 94% | 100% / 100% | 588 / 1337 | +0.546 |
| 2022 | 3,169 | 100% / 92% | 100% / 91% | 100% / 100% | 0% / 0% | 353 / 306 | n/a |
| 2024 | 3,421 | 0% / 94% | 0% / 90% | 0% / 100% | 0% / 100% | 1 / 575 | +0.714 |

Coverage is the share of rows with a substantive value after restricting
each item to its valid range (pid7 1--7, ideo5 1--5, educ 1--6, lcself 1--7).
`Profiles` counts distinct education/political/alignment/distance blocks
rendered by `prompts.build_candidate_prompt`.

## Mapping

- 2018: use `pid7x`; the `pid7` column is a placeholder (`-4`).
- 2020: `pid7` is reverse-coded; use `8 - pid7`. `ideo5` is not in the raw
  file; derive it from `lcself`.
- 2024: raw `pid7`, `ideo5`, and `educ` are text labels; map them to numeric
  codes. `lcself` comes from `ideology_lcself`.
- 2016/2019/2022: v3 values restricted to valid ranges.

The row-level base table is `anes_candidates_base_v4cov.csv`; it is not a
training table until the interaction features are recomputed.
