# Raw ANES audit

- rows: **132,948**
- respondents: **3,165**
- respondents in >1 year: **3,080**
- max years per respondent: **6**
- train-year vs test-year respondent overlap: **1,909**
- test-only respondents: **0**

## Candidate name issues

- trailing dot rows: **8,165**
- whitespace rows: **0**
- uppercase rows: **0**
- group-like targets mixed in: **23,060**

## Target

- missing `support_score`: **1,550**
- out of range: **2,106**

## Duplicates

- rows sharing `(respondent_id, year, candidate)`: **0**
- duplicate groups: **0**

## Rows per year

| year | rows |
|---|---|
| 2016 | 13,200 |
| 2018 | 22,500 |
| 2019 | 31,650 |
| 2020 | 55,440 |
| 2022 | 6,340 |
| 2024 | 3,818 |

## Candidates per year (raw)

- **2016** (11): black, carson, cruz, fiorina, hrc, jeb, obama, rubio, sanders, trump, white
- **2018** (9): black, hrc, hrc., kavanaugh, mueller, obama, obama., trump, white
- **2019** (10): biden, black, buttigieg, harris, obama, obama., sanders, trump, warren, white
- **2020** (18): biden, black, buttigieg, fauci, haley, harris, klobuchar, obama, ocasioc, pelosi, pence, rubio, sanders, thomas, trump, warren, white, yang
- **2022** (4): biden, dem, rep, trump
- **2024** (2): biden, trump

## Candidates per year (normalized, groups removed)

- **2016** (9): carson, cruz, fiorina, hrc, jeb, obama, rubio, sanders, trump
- **2018** (5): hrc, kavanaugh, mueller, obama, trump
- **2019** (7): biden, buttigieg, harris, obama, sanders, trump, warren
- **2020** (16): biden, buttigieg, fauci, haley, harris, klobuchar, obama, ocasioc, pelosi, pence, rubio, sanders, thomas, trump, warren, yang
- **2022** (2): biden, trump
- **2024** (2): biden, trump