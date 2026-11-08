## 3 Data Construction

### 3.1 Source surveys

We build the individual-level corpus from the American National Election Studies (ANES) pilot waves of 2016, 2018, 2019, 2020, 2022, and 2024. The raw respondent-by-target table contains 132,948 rows, 3,165 unique respondents, and 29 candidate targets. For the population-level component we use the General Social Survey (GSS) waves of 2004, 2008, 2012, 2014, 2016, 2018, 2021, and 2022, with 2024 as the target year.

### 3.2 Cleaning

The raw ANES feature table contains two defects that materially affect any downstream evaluation. First, several feeling-thermometer items are stored twice with a trailing period (`obama.` and `obama`, `hrc.` and `hrc`), producing 8,165 duplicate rows. Second, social-group thermometers (`black`, `white`, `dem`, `rep`) are stored in the same column as candidate thermometers, producing 23,060 rows in which the "candidate" is in fact a group. Treating these rows as candidate preferences contaminates both training and evaluation.

We therefore normalize target names by lower-casing and stripping trailing periods, separate candidate targets from group targets, drop rows with missing or out-of-range scores (3,588), and deduplicate on `(respondent, year, target)` (8,141 rows). The resulting candidate table contains **98,159** interaction-level rows. We retain the original row index so that every cleaned row can be traced back to the raw table. Table 1 summarizes the audit.

### 3.3 Interaction-level representation

A wide-format ANES row stores one voter and many thermometer columns, which implicitly treats a rating as a property of the voter and suppresses voter-target interactions. We instead represent each observation as a triple

$$
(v_i,\ c_j,\ y_{ij}\mid t_i),
$$

where $v_i$ is the voter profile, $c_j$ the target, $y_{ij}\in[0,100]$ the feeling thermometer, and $t_i$ the survey year. This yields one row per voter-target pair and makes the interaction structure explicit.

Candidate identity is decoupled from candidate attributes. Each candidate is represented by a metadata vector

$$
\mathcal{M}[c] = (\text{party},\ \text{ideology},\ \text{age},\ \text{position},\ \text{incumbent}),
$$

so that the model learns $P(y\mid v,\mathcal{M}[c],t)$ rather than memorizing a candidate name. The same transformation is applied to social groups: a group is described by its category (race, party, ideology, or social movement), ideological position, and valence. The group corpus contains 61,710 voter-group pairs over nine groups (Black, White, Hispanic, and Asian Americans; the Democratic and Republican parties; Socialists and Capitalists; and Feminists).

### 3.4 Social-cognitive features

On top of the interaction-level rows we compute a compact set of features that encode theoretical primitives of political cognition. Partisan reasoning is captured by

$$
\text{party\_match}(v,c)=
\begin{cases}
+1 & \text{if } \text{party}_v=\text{party}_c,\\
-1 & \text{if } v \text{ is an independent},\\
0 & \text{otherwise}.
\end{cases}
$$

Spatial reasoning is captured by the seven-point ideological distance

$$
\text{ideo\_dist}(v,c)=\lvert \text{lcself}_v-\text{ideology}_c\rvert\in[0,6],
$$

and a composite similarity index combines party, ideology, race, gender, and age:

$$
\text{sim}(v,c)=0.25\,\mathbb{1}[\text{party match}]
+0.25\,\mathbb{1}[\text{same ideological side}]
+0.20\,\mathbb{1}[\text{race match}]
+0.15\,\mathbb{1}[\text{gender match}]
+0.15\left(1-\frac{\lvert \text{age}_v-\text{age}_c\rvert}{100}\right).
$$

Finally, cross-pressure indicators flag cases in which partisan and ideological cues conflict:

$$
\begin{aligned}
\text{CP}_1 &= \mathbb{1}[\text{party match}=1 \land \text{ideo\_dist}\ge 3],\\
\text{CP}_2 &= \mathbb{1}[\text{party match}=0 \land \text{ideo\_dist}\le 1],\\
\text{CP}_3 &= \mathbb{1}[\text{sim}\ge 0.7 \land \text{party match}=0].
\end{aligned}
$$

These features are not arbitrary predictors: they operationalize partisan identity, spatial proximity, and ambivalence, and they are the same features we later give to the tabular baseline.

### 3.5 Distribution-aware sampling

Uniform sampling from a survey over-represents common profiles and under-represents the tails and conflict cases that carry most of the theoretical interest. We therefore draw the training pool with nine complementary strategies: balanced coverage of the score range, hard cases near the middle of the scale, party-by-ideology cells, similarity-stratified bins, cross-pressure cases, extreme scores, incumbent effects, temporal coverage, and candidate-specific balance. The resulting pool contains 6,516 rows and is intentionally not a uniform sample; the test sets, in contrast, are built from all cleaned rows in the target years so that evaluation is not shaped by the same sampling design.

### 3.6 Evaluation protocols and the panel caveat

ANES is a panel: 3,080 of the 3,165 respondents appear in more than one wave, and every respondent observed in 2022 or 2024 also appears in 2016--2020. There are **zero** test-only respondents. A temporal split can therefore measure transfer to a later wave, but it cannot isolate generalization to unseen individuals.

We report two complementary protocols. The **temporal protocol (T)** trains on 2016--2020 (6,516 rows), validates on 2022 (2,501), and tests on 2024 (3,421). The **respondent protocol (R)** splits respondents 80/20 by a hash of the respondent identifier, independent of year: 5,094 training rows and 6,014 test rows. We also report a strict cell, **T×R**, containing the 226 rows from 2024 whose respondents are in the 20% held-out set. Within each protocol the splits are disjoint; the two protocols intentionally overlap across each other. R is our main in-distribution test, while T measures the temporal boundary.

For groups, coverage is uneven across waves (three groups in 2016, six in 2018, four in 2019, nine in 2020, two in 2022), so a temporal group split is not meaningful. We instead use a within-2020 block split (80/20 respondents, all nine groups) and a multi-year respondent split over the six groups with reliable identifiers.

### 3.7 Population-level survey data

For the population-level task we reconstruct the joint distribution of three policy attitudes from GSS. We use two configurations: **EDC** (education, drug addiction, crime) and **EHC** (environment, health, cities). Each attribute takes three levels (`too little`, `about right`, `too much`). Eight historical waves provide the time series; the target is 2024. Section 4.5 describes how the joint distribution is extrapolated and used to build fine-tuning tasks.
