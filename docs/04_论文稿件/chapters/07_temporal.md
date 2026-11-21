## 7 Temporal Analysis

### 7.1 Setup

The temporal protocol (T) trains on 2016--2020 and tests on 2024. We also report a per-year view of the temporal-protocol model on the respondent-protocol rows: for each wave from 2016 to 2024, we evaluate the model on the rows contributed by the 20% hash-selected respondent numbers. The temporal model is trained on 2016--2020 for all respondents, so this view does not hold out respondent numbers; it isolates the year while holding the respondent pool fixed. The 2024 wave is additionally evaluated on the full 3,421-row test set (all respondents), and we compare with HistGB trained on the same features. The R models are evaluated separately on 2024 (Section 7.3).

### 7.2 The decay is monotonic and steep

Table 3 and Figure 2 report the per-year results. For the fine-tuned LLM, the correlation with the ground truth is 0.722 in 2019, 0.478 in 2020, 0.408 in 2022, and $-$0.056 in 2024; MAE rises from 19.20 to 25.39, 27.76, and 37.30 over the same waves. HistGB shows the same direction but decays later: 0.746 in 2019, 0.560 in 2020, 0.657 in 2022, and $-$0.008 in 2024; MAE rises from 18.85 to 23.41, 27.98, and 35.52.

Two features of this pattern matter. First, the decay is not a sudden break at 2024; performance drops between 2019 and 2020. But 2020 is in the training window, and its drop is not a clean wave effect: 2020 has no ideology item and a reversed partisanship scale (Section 9), so the model receives contradictory political inputs; the temporal boundary proper appears in 2022 and 2024. Second, the LLM decays earlier than the tabular baseline: by 2022 its correlation has fallen to 0.408 while HistGB is still at 0.657. A model that relies on a learned representation of partisan and ideological structure is apparently more sensitive to a changing political context than a model that relies on a small set of explicit features.

### 7.3 The 2024 collapse

On the full 2024 test set the collapse is complete. The fine-tuned LLM reaches an MAE of 36.47 $\pm$ 0.75 but a correlation of $-$0.029 $\pm$ 0.021; HistGB reaches 35.06 and $-$0.030. A constant prediction of 50 reaches an MAE of 35.05 on the same rows, and a candidate-mean baseline reaches 34.84; the LLM's and HistGB's aggregate MAEs are therefore at or above the no-signal baseline. The MAE should not be read as evidence of individual-level signal. The predictions concentrate on the middle of the scale: 3,173 of 3,421 LLM predictions are exactly 50, and every Biden row is predicted as 50 (standard deviation 0.0), while Trump rows have a standard deviation of 17.4 and a correlation of $-$0.014. The anchor readout does not rescue the signal: on the same rows it produces a correlation of 0.003 with a prediction standard deviation of 2.1. The low ECE of the anchor readout (2.90) is therefore an artifact of near-constant predictions, not evidence of calibration.

This is not an LLM-specific failure. HistGB, which has no language-model component, also loses the individual signal in 2024. The R models show the same collapse: on the full 2024 test set their correlation is $-$0.02 to $-$0.06, and on the strict 226-row T×R cell it is $-$0.09 to $-$0.15. The comparison is nonetheless confounded by covariate availability: all 3,421 prompts for 2024 render the same political profile, because party identification, ideology, and education are absent from that wave. The 2024 evaluation therefore measures what both models do when the political covariates are removed, not only what four years of political change do.

A covariate-dropout probe on the 2019 held-out rows replaces the political block with the 2024 defaults (Independent, Moderate, High school, Independent voter, distance 0.0). LLM correlation falls from 0.714 to 0.166 and HistGB from 0.734 to 0.172; MAE rises from 19.34 to 31.67 and 19.04 to 31.46. The actual 2024 values are $-0.055$/$+0.015$ and 37.78/35.11. Missing covariates therefore explain most, but not all, of the collapse; a smaller residual year or candidate-set shift remains.

### 7.4 Adding the most recent wave does not help

A natural mitigation is to include the most recent available wave in training. We fine-tune the same model on the union of the 2016--2020 pool and the 2022 validation rows (9,017 rows) and evaluate on 2024. The result is essentially unchanged: MAE 36.62, correlation $-$0.030, choice 0.525, and a prediction standard deviation of 20.1. Adding 2022 does not restore the individual signal. This rules out a simple "the model has not seen recent data" explanation and points to a residual distribution shift that one additional wave does not cover, after the input-availability effect above is accounted for.

### 7.5 Interpretation

The temporal results support C3 in a strong form. Both the LLM and the tabular baseline learn a mapping that is valid for the 2016--2019 era but fails four years later. The LLM's earlier decay suggests that its representation is more tightly coupled to the training-era political structure; the tabular model's explicit features are more robust for one additional wave but fail in the same way at 2024. The practical implication is that a survey simulator trained on historical waves should not be used for a future election without a temporal validation protocol and, ideally, a domain-adaptation step. Reporting a single aggregate MAE over a pooled 2016--2024 test set would hide the collapse, because the pooled number is dominated by the earlier waves.

We therefore recommend that temporal shift be treated as a first-class evaluation axis for LLM survey simulation, alongside individual accuracy and distributional alignment. The negative result is informative: structured alignment is necessary but not sufficient for temporal generalization.

### 7.6 Relation to the population-level results

The FAS results in Section 6 are computed for a fixed target year with a reconstructed joint distribution, not for an unseen future wave. They therefore do not suffer from the same temporal shift, but they also do not demonstrate temporal extrapolation. The two components answer different questions: PAS asks whether the model can act as an individual respondent, FAS asks whether it can reproduce a population distribution. Neither, in our experiments, supports extrapolation to a substantially different political era.
