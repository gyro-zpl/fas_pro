## 5 PAS Experiments

### 5.1 Setup

We fine-tune Qwen3-8B with LoRA on the 6,516-row training pool described in Section 3.5, using three seeds (42, 43, 44). The main in-distribution evaluation uses the respondent protocol (R) restricted to the 2016, 2018, and 2019 waves: 2,726 rows from respondents never seen during training. The 2020 wave is treated as a near-domain set, and 2022/2024 form the temporal test analyzed in Section 7. We report MAE, Pearson correlation, rating-induced choice accuracy, and chance accuracy. Baselines are HistGB and Ridge on the same tabular features, plus the base model with the same prompts but no fine-tuning. All confidence intervals are bootstrap 95% intervals over rows (MAE, correlation) or respondent-year groups (choice).

### 5.2 In-distribution results

Table 2 reports the main result. The fine-tuned LLM reaches an MAE of 20.84 [20.38, 21.30], a correlation of 0.679 [0.666, 0.692], and a choice accuracy of 0.518 [0.481, 0.552] against a chance level of 0.163. HistGB reaches 20.58 [20.01, 21.12], 0.714 [0.696, 0.733], and 0.534 [0.498, 0.569]; Ridge is weaker at 22.47, 0.655, and 0.484.

The honest reading is that the LLM is **competitive with, but not superior to, a strong tabular baseline**. A paired bootstrap over the same 2,726 rows confirms this: for seed 42 the LLM--GBM difference in MAE is $+0.60$ $[-0.19, +1.42]$ (seeds 43 and 44: $+0.02$ and $+0.19$), and the difference in choice is $+0.026$ $[-0.007, +0.062]$; both intervals include zero. The correlation difference is $-0.039$ $[-0.056, -0.021]$, consistently negative across seeds. Fine-tuning, by contrast, improves MAE by 2.98 $[2.10, 3.82]$ relative to the base model, with a correlation gain of $+0.082$ and a choice gain of $+0.081$. We therefore do not claim that the language model is a better individual-level predictor than feature-engineered regression. What the LLM adds is different: a single model that consumes the same survey vignette can produce a full rating distribution (Section 4.3) and can be extended to population-level generation (Section 6), whereas the tabular baselines cannot.

The gap to the prompt-only model is large. On the same in-distribution rows, the base LLM reaches an MAE of 24.16 [23.44, 24.88], a correlation of 0.593, and a choice accuracy of 0.479; on the 2020 wave it degrades to an MAE of 44.89 and a correlation of $-$0.389. Fine-tuning is therefore necessary for the model to behave like a survey respondent rather than a generic assistant (Table 2, last row).

### 5.3 Per-year behavior

Table 3 and Figure 2 break the in-distribution result down by wave. Performance is best in 2019 (MAE 19.20, correlation 0.722, choice 0.663 against chance 0.150) and weaker in 2016 (22.45, 0.588, 0.329 against 0.112). HistGB shows the same ordering, with a higher correlation in all three waves (0.611, 0.729, 0.746) and a similar choice profile (0.256, 0.544, 0.639). The LLM narrows the gap in 2019 but does not overtake the baseline.

This pattern is consistent with the interpretation that both models recover the same partisan and ideological structure; the LLM's advantage, if any, is not in average accuracy but in the form of its output. Figure 4 shows the corresponding density: predictions are correlated with the ground truth but are compressed toward the middle of the scale, with a visible band at 50. The compression is a property of the fine-tuned readout, not of the correlation alone, and it motivates the anchor readout discussed next.

### 5.4 Generation versus anchor readout

The two readouts in Section 4.3 behave differently. On the full 2024 test set (seed 42), the anchor readout gives a lower MAE than generation (34.99 versus 35.79) and a much lower ECE (2.90 versus 10.46), while choice accuracy is similar (0.547 versus 0.536). However, the anchor predictions have a standard deviation of only 2.1 and a correlation with the truth of 0.003: the low ECE is produced by an almost constant prediction near 40, not by genuine calibration. Generation has a larger spread (standard deviation 12.3) and a correlation of $-$0.009 on the same rows. Neither readout recovers individual signal in 2024.

On the in-distribution years, where predictions do vary, generation is the primary readout because it preserves individual differences and yields the choice results above. The anchor readout is useful as a diagnostic of the model's rating distribution, but it should not be reported as evidence of calibration unless the prediction spread is reported alongside it. We return to this point in Section 7, where the 2024 collapse is analyzed as a temporal rather than a readout failure.

### 5.5 Discussion

Three conclusions follow. First, structured fine-tuning is necessary: the base model is both biased and unstable, and its 2020 correlation is negative. Second, the fine-tuned model is a credible individual-level simulator in-distribution, but it does not beat a feature-engineered GBM; the contribution of the language model must therefore be argued on other grounds. Third, the readout matters: a low ECE can be an artifact of prediction collapse, so calibration claims must be paired with correlation and spread.

The results support a narrow form of C1: structured survey data alignment gives a compact LLM genuine individual-level signal on unseen respondents in the same era. They do not support a claim of LLM superiority over tabular baselines, and we do not make one.
