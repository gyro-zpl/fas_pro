## 5 PAS Experiments

### 5.1 Setup

We fine-tune Qwen3-8B with LoRA on the `train_r` pool (5,094 rows from 2,131 respondents) described in Section 3.5, using three seeds (42, 43, 44). The main in-distribution evaluation uses the respondent protocol (R) restricted to the 2016, 2018, and 2019 waves: 2,726 rating rows from the 20% hash holdout, whose respondent numbers do not appear in `train_r`. The 2020 wave is treated as a near-domain set, and 2022/2024 form the temporal test analyzed in Section 7. We report MAE, Pearson correlation, rating-induced choice accuracy, and chance accuracy. Baselines are HistGB and Ridge trained on `train_r` on the same tabular features, plus the base model with the same prompts but no fine-tuning. All confidence intervals are bootstrap 95% intervals (Efron and Tibshirani, 1993) over rows (MAE, correlation) or respondent-year groups (choice).

### 5.2 In-distribution results

Table 2 reports the main result. The fine-tuned LLM reaches an MAE of 21.35 [20.87, 21.83], a correlation of 0.666 [0.652, 0.680], and a choice accuracy of 0.525 [0.478, 0.573] against a chance level of 0.163. HistGB reaches 20.73 [20.16, 21.26], 0.703 [0.684, 0.724], and 0.543 [0.495, 0.591]; Ridge is weaker at 22.86, 0.639, and 0.450. A second checkpoint, DeepSeek-R1-Distill-Llama-8B, trained on `train_r` with one seed, reaches 21.37, 0.665, and 0.572; the in-distribution signal is therefore not specific to Qwen3-8B.

The honest reading is that the LLM is **competitive with, but not superior to, a strong tabular baseline**. A paired bootstrap over the same 2,726 rows confirms this: the LLM--GBM MAE difference is $+0.59$ $[-0.17, +1.32]$ for seed 42, $+0.42$ $[-0.34, +1.19]$ for seed 43, and $+0.89$ $[+0.10, +1.67]$ for seed 44; the difference is small, and the baseline is never worse. The argmax choice difference is mixed ($+0.026$, $-0.028$, and $-0.052$), while tie-aware choice consistently favors the baseline ($-0.057$, $-0.082$, and $-0.083$). The correlation difference is consistently negative ($-0.040$, $-0.033$, and $-0.039$). Fine-tuning, by contrast, improves MAE by 2.5--3.0 relative to the base model, with a correlation gain of about $+0.07$ and a choice gain of $+0.01$ to $+0.09$ depending on seed. We therefore do not claim that the language model is a better individual-level predictor than feature-engineered regression. What the LLM adds is different: a single model that consumes the same survey vignette can produce a full rating distribution (Section 4.3) and can be extended to population-level generation (Section 6), whereas the tabular baselines cannot.

The gap to the prompt-only model is large. On the same in-distribution rows, the base LLM reaches an MAE of 24.15 [23.44, 24.89], a correlation of 0.594, and a choice accuracy of 0.478; on the 2020 wave it degrades to an MAE of 44.89 and a correlation of $-$0.389. Fine-tuning is therefore necessary for the model to behave like a survey respondent rather than a generic assistant (Table 2, last row).

### 5.3 Per-year behavior

Table 3 and Figure 2 break the result down by wave for the temporal-protocol model on the held-out respondent rows; the respondent-held-out models show the same ordering with slightly higher MAE. Performance is best in 2019 (MAE 19.20, correlation 0.722, choice 0.663 against chance 0.150) and weaker in 2016 (22.45, 0.588, 0.329 against 0.112). HistGB shows the same ordering, with a higher correlation in all three waves (0.611, 0.729, 0.746) and a similar choice profile (0.256, 0.544, 0.639). The LLM narrows the gap in 2019 but does not overtake the baseline.

This pattern is consistent with the interpretation that both models recover the same partisan and ideological structure; the LLM's advantage, if any, is not in average accuracy but in the form of its output. Figure 4 shows the corresponding density: predictions are correlated with the ground truth but are compressed toward the middle of the scale, with a visible band at 50. The compression is a property of the fine-tuned readout, not of the correlation alone, and it motivates the anchor readout discussed next.

### 5.4 Generation versus anchor readout

The two readouts in Section 4.3 behave differently. On the full 2024 test set (seed 42), the anchor readout gives a lower MAE than generation (34.99 versus 35.79) and a much lower ECE (2.90 versus 10.46), while choice accuracy is similar (0.547 versus 0.536). However, the anchor predictions have a standard deviation of only 2.1 and a correlation with the truth of 0.003: the low ECE is produced by an almost constant prediction near 40, not by genuine calibration. Generation has a larger spread (standard deviation 12.3) and a correlation of $-$0.009 on the same rows. Neither readout recovers individual signal in 2024.

On the in-distribution years, where predictions do vary, generation is the primary readout because it preserves individual differences and yields the choice results above. The anchor readout is useful as a diagnostic of the model's rating distribution, but it should not be reported as evidence of calibration unless the prediction spread is reported alongside it. We return to this point in Section 7, where the 2024 collapse is analyzed as a temporal rather than a readout failure.

### 5.5 Discussion

Three conclusions follow. First, structured fine-tuning is necessary: the base model is both biased and unstable, and its 2020 correlation is negative. Second, the fine-tuned model is a credible individual-level simulator in-distribution, but it does not beat a feature-engineered GBM; the contribution of the language model must therefore be argued on other grounds. Third, the readout matters: a low ECE can be an artifact of prediction collapse, so calibration claims must be paired with correlation and spread.

The results support a narrow form of C1: structured survey data alignment gives a compact LLM genuine individual-level signal on held-out respondents in the same era. They do not support a claim of LLM superiority over tabular baselines, and we do not make one.
