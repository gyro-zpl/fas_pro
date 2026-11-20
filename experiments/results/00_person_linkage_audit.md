# Person-level linkage audit (2026-11-20)

## Raw ANES files

| Year | Rows | Unique IDs | ID min | ID max | ID kind |
|---|---:|---:|---:|---:|---|
| 2016 | 1200 | 1200 | 1.0 | 1200.0 | integer |
| 2018 | 2500 | 2500 | 1.0 | 2779.0 | integer |
| 2019 | 3165 | 3165 | 1.0 | 3165.0 | integer |
| 2020 | 3080 | 3080 | — | — | random string |
| 2022 | 1585 | 1585 | 1.0 | 1585.0 | integer |
| 2024 | 1909 | 1909 | 1.0 | 1909.0 | integer |

## Cross-wave overlap of the raw caseid

| Pair | Overlap IDs | Birth-year match | Gender match |
|---|---:|---:|---:|
| 2016–2018 | 1092 | 0.0174 | 0.4936 |
| 2016–2019 | 1200 | 0.0258 | 0.5167 |
| 2016–2022 | 1200 | 0.0167 | 0.4925 |
| 2016–2024 | 1200 | 0.0167 | 0.0000 |
| 2018–2019 | 2500 | 0.0196 | 0.5132 |
| 2018–2022 | 1417 | 0.0148 | 0.4778 |
| 2018–2024 | 1695 | 0.0136 | 0.0000 |
| 2019–2022 | 1585 | 0.0183 | 0.4795 |
| 2019–2024 | 1909 | 0.0115 | 0.0000 |
| 2022–2024 | 1585 | 0.0158 | 0.0000 |

A genuine cross-wave person ID would keep the same birth year and gender for the same ID. The near-chance agreement shows that the same `caseid` in two waves is a different person.

## Processed analysis table

The clean table uses a 0-based within-wave `respondent_id`; cross-wave number overlap has near-chance demographic agreement:

| Pair | Overlapping numbers | Birth-year match | Gender match |
|---|---:|---:|---:|
| 2016–2018 | 1200 | 0.0162 | 0.5150 |
| 2016–2019 | 1198 | 0.0263 | 0.5173 |
| 2016–2020 | 1198 | 0.0084 | 0.0000 |
| 2016–2022 | 1200 | 0.0000 | 0.4933 |
| 2016–2024 | 1083 | 0.0176 | 0.0000 |
| 2018–2019 | 2496 | 0.0191 | 0.5155 |
| 2018–2020 | 2498 | 0.0128 | 0.0000 |
| 2018–2022 | 1585 | 0.0000 | 0.5133 |
| 2018–2024 | 1712 | 0.0124 | 0.0000 |
| 2019–2020 | 3070 | 0.0177 | 0.0000 |
| 2019–2022 | 1583 | 0.0000 | 0.4801 |
| 2019–2024 | 1710 | 0.0123 | 0.0000 |
| 2020–2022 | 1583 | 0.0000 | 0.0000 |
| 2020–2024 | 1710 | 0.0143 | 0.0000 |
| 2022–2024 | 1413 | 0.0000 | 0.0000 |

## v3 split number overlap (not person-level)

**R protocol (train_r -> test_r)**

- train rows: 5094, numbers: 2131
- test rows: 6014, numbers: 192
- test numbers seen in train: 0
- test rows from seen numbers: 0
- in-distribution rows: 2726; from seen numbers: 0

**T protocol (train_t -> test_r)**

- train rows: 6516, numbers: 2720
- test rows: 6014, numbers: 192
- test numbers seen in train: 170
- test rows from seen numbers: 5443
- in-distribution rows: 2726; from seen numbers: 2492

The R split is disjoint in respondent **numbers** by construction (0 overlap); this is the 'held-out respondent numbers' claim. It does not establish person-level isolation, because the same person can appear under different within-wave numbers. The T split reuses test numbers in training by design and is not respondent-held-out. Neither overlap is a person-level leakage measurement.

## Conclusion

The released raw CSV caseid is a within-wave 1..N number, not a cross-wave person key; the processed respondent_id is 0..N-1. Cross-wave birth-year agreement is near chance (~2%), and 2020 only carries a random ResponseId. Person-level v4 splits cannot be built from these files. Fallback B applies: report held-out respondent numbers within a wave, and state that person-level isolation is unverified.

If the original ANES `.dta`/`.sav` files with a stable panel identifier (e.g., `V160001`/`V200001`/`V240001`) become available, the v4 person-level rerun can proceed. Until then, the paper should use 'held-out respondent numbers within a wave' and list person-level isolation as unverified.