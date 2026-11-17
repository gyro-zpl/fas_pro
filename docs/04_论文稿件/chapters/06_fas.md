## 6 FAS Experiments

### 6.1 Setup

The population-level evaluation uses the GSS data described in Section 3.7. For each of the two three-attribute configurations (EDC and EHC), we reconstruct the 2024 joint distribution from eight historical waves, sample 400 synthetic respondents, and compare the generated sample with the empirical 2024 sample (1,552 respondents for EHC and 1,601 for EDC). We report Jensen--Shannon (JS) divergence on the three marginals and the three pairwise joints, averaged within each family. Lower is better. Symmetric KL is reported as a secondary metric. As a reference, we also draw 400 rows directly from the reconstructed joint $Q^{\ast}$ without a language model (the prior-only baseline).

We compare six generation conditions: the base model, direct supervised fine-tuning (SFT), retrieval-augmented generation without fine-tuning (RAG), RAG plus SFT, a second SFT run with a different task mixture (v2), a third SFT run (v3), and a quota-annealing sampler. All generation runs use the same prompts and the same target year; only the training or sampling procedure changes.

### 6.2 Main results

Table 4 and Figure 3 summarize the results. The prior-only baseline reaches marginal and joint JS of 0.0004 and 0.0076 on EDC and 0.0011 and 0.0225 on EHC, lower than every LLM condition. The LLM results are therefore movement toward $Q^{\ast}$: on EDC, the base model produces 0.0552 and 0.1181; direct SFT reduces these to 0.0048 and 0.0160 (91% and 86%), and v3 reaches 0.0040 and 0.0215 (93% and 82%). On EHC, the base model starts at 0.0319 and 0.1001; direct SFT reaches 0.0167 and 0.0470 (48% and 53%), and v3 reaches 0.0077 and 0.0363 (76% and 64%). Fine-tuning moves the LLM substantially toward $Q^{\ast}$, but it does not surpass $Q^{\ast}$: the prior remains better on both marginal and joint JS in both configurations. The large-reduction claim is scoped to the LLM-versus-base comparison.

Bootstrap 95% CIs over the generated or prior-sampled rows separate the prior from v3 on the joint metric (EDC 0.0076 [0.0070, 0.0141] versus 0.0215 [0.0181, 0.0305]; EHC 0.0225 [0.0192, 0.0308] versus 0.0363 [0.0332, 0.0458]). The EHC result is weaker than EDC: the marginal reduction is large in the best run, but the joint reduction is only about two thirds. We report both configurations rather than selecting the favorable one.

The two metrics do not always agree. On EHC, the symmetric KL of the joint distribution increases from 0.4289 (base) to 0.5487 (direct SFT) even though JS falls from 0.1001 to 0.0470. KL is sensitive to near-zero cells and to the 400-sample size; JS is bounded and more stable. We therefore treat JS as the primary metric and KL as a diagnostic, and we do not claim a KL improvement for direct SFT on EHC. The v3 run improves both JS and KL (0.0363 and 0.3064 versus 0.1001 and 0.4289), which is why v3 is the EHC reference run.

### 6.3 Component analysis

Three comparisons isolate the source of the gain.

**RAG is not the driver.** RAG without fine-tuning is close to the base model on EDC (0.0473 marginal, 0.1096 joint) and slightly worse than the base on EHC (0.0404, 0.1095). Adding retrieval does not by itself teach the model the target distribution.

**SFT drives the movement toward $Q^{\ast}$.** RAG plus SFT performs like direct SFT on both configurations (EDC 0.0054/0.0176; EHC 0.0152/0.0427). The movement comes from supervised exposure to the reconstructed distribution and the five task types, not from retrieval.

**Quota annealing is not better.** The quota sampler reaches 0.0091/0.0216 on EDC and 0.0175/0.0522 on EHC, worse than direct SFT on both; it also did not converge within 15 rounds, so the comparison is descriptive. We therefore do not use quota annealing in the main method.

### 6.4 Why EDC and EHC differ

EDC and EHC are not equally predictable. EDC (education, drugs, crime) has stronger and more stable temporal trends in the GSS waves, so the extrapolated 2024 joint is closer to the truth and the fine-tuning signal is cleaner. EHC (environment, health, cities) is more volatile and less correlated across attributes; the ensemble extrapolation is less accurate, and the joint reconstruction has more room for error. We treat this as an empirical difference between configurations rather than as evidence that the method is uniformly effective.

### 6.5 Sample completeness

The generation conditions differ in how many valid rows they produce. The base model returns 351 of 400 requested rows on EHC (coverage 0.877), while the SFT runs return 399--400; on EDC, direct SFT returns 381 rows (coverage 0.953) and the v3 run returns 400. Because the distributional metrics are computed on the returned rows, a condition that produces fewer rows is not penalized for the missing mass. To make the comparison conservative, we charge every missing row as maximally wrong (the maximum JS for a categorical distribution is $\ln 2$). Under this adjustment, the base model's EHC marginal and joint JS rise from 0.0319/0.1001 to 0.1168/0.1850, while the fully covered v3 run stays at 0.0077/0.0363. On EDC, direct SFT's raw 0.0048/0.0160 becomes 0.0377/0.0489 under the same penalty, whereas v3, which returns all 400 rows, is unaffected at 0.0040/0.0215.

We therefore treat v3 as the cleanest LLM result: it achieves full coverage on both configurations and the lowest raw JS among the LLM runs. Direct SFT still shows a large improvement over the base model, but part of its apparent advantage on EDC is due to the 19 missing rows. We report raw and coverage-adjusted values in Table 4 and recommend that future work report both.

### 6.6 Discussion

The FAS results support a boundary for C2: supervised fine-tuning on a reconstructed joint distribution moves the LLM toward the prior by roughly an order of magnitude relative to the base model, and the movement is not explained by retrieval or quota sampling. But direct sampling from the prior is more accurate than every LLM condition, so the LLM does not beat the classical distributional baseline. For EHC the LLM's gain over its base is smaller, and the KL diagnostic moves in the opposite direction for direct SFT. The honest summary is that the LLM approaches the prior when the underlying temporal trends are stable and the attributes are correlated, but it does not replace the prior.

The language model's distinctive capability is therefore not unconditional distribution matching; it is the ability to generate a synthetic population and to condition, perturb, or correct that population in natural language, which the current evaluation does not isolate. The individual-level results in Section 5 are competitive rather than superior; the population-level results establish a boundary rather than a win.
