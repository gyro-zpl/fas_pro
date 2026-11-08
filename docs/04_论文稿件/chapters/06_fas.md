## 6 FAS Experiments

### 6.1 Setup

The population-level evaluation uses the GSS data described in Section 3.7. For each of the two three-attribute configurations (EDC and EHC), we reconstruct the 2024 joint distribution from eight historical waves, sample 400 synthetic respondents, and compare the generated sample with the empirical 2024 sample (1,552 respondents for EHC and 1,601 for EDC). We report Jensen--Shannon (JS) divergence on the three marginals and the three pairwise joints, averaged within each family. Lower is better. Symmetric KL is reported as a secondary metric.

We compare six generation conditions: the base model, direct supervised fine-tuning (SFT), retrieval-augmented generation without fine-tuning (RAG), RAG plus SFT, a second SFT run with a different task mixture (v2), a third SFT run (v3), and a quota-annealing sampler. All generation runs use the same prompts and the same target year; only the training or sampling procedure changes.

### 6.2 Main results

Table 4 and Figure 3 summarize the results. On EDC, the base model produces a marginal JS of 0.0552 and a joint JS of 0.1181. Direct SFT reduces these to 0.0048 and 0.0160, a relative reduction of **91% and 86%**. The best SFT run (v3) reaches 0.0040 and 0.0215, i.e., **93% and 82%**. On EHC, the base model starts at 0.0319 and 0.1001; direct SFT reaches 0.0167 and 0.0470 (48% and 53%), and v3 reaches 0.0077 and 0.0363 (**76% and 64%**).

The EDC result supports the claim that fine-tuning substantially improves population-level distributional alignment, including the pairwise structure that a marginal-only sampler would miss. The EHC result is weaker: the marginal reduction is large in the best run, but the joint reduction is only about two thirds. We report both configurations rather than selecting the favorable one, and we scope the "80% reduction" claim to EDC.

The two metrics do not always agree. On EHC, the symmetric KL of the joint distribution increases from 0.4289 (base) to 0.5487 (direct SFT) even though JS falls from 0.1001 to 0.0470. KL is sensitive to near-zero cells and to the 400-sample size; JS is bounded and more stable. We therefore treat JS as the primary metric and KL as a diagnostic, and we do not claim a KL improvement for EHC.

### 6.3 Component analysis

Three comparisons isolate the source of the gain.

**RAG is not the driver.** RAG without fine-tuning is close to the base model on EDC (0.0473 marginal, 0.1096 joint) and slightly worse than the base on EHC (0.0404, 0.1095). Adding retrieval does not by itself teach the model the target distribution.

**SFT is the driver.** RAG plus SFT performs like direct SFT on both configurations (EDC 0.0054/0.0176; EHC 0.0152/0.0427). The improvement comes from supervised exposure to the reconstructed distribution and the five task types, not from retrieval.

**Quota annealing is not better.** The quota sampler reaches 0.0091/0.0216 on EDC and 0.0175/0.0522 on EHC, worse than direct SFT on both. We therefore do not use quota annealing in the main method.

### 6.4 Why EDC and EHC differ

EDC and EHC are not equally predictable. EDC (education, drugs, crime) has stronger and more stable temporal trends in the GSS waves, so the extrapolated 2024 joint is closer to the truth and the fine-tuning signal is cleaner. EHC (environment, health, cities) is more volatile and less correlated across attributes; the ensemble extrapolation is less accurate, and the joint reconstruction has more room for error. We treat this as an empirical difference between configurations rather than as evidence that the method is uniformly effective.

### 6.5 Sample completeness

The generation conditions differ in how many valid rows they produce. The base model returns 351 of 400 requested rows on EHC (coverage 0.877), while the SFT runs return 399--400; on EDC, direct SFT returns 381 rows (coverage 0.953) and the v3 run returns 400. Because the distributional metrics are computed on the returned rows, a condition that produces fewer rows is not penalized for the missing mass. To make the comparison conservative, we charge every missing row as maximally wrong (the maximum JS for a categorical distribution is $\ln 2$). Under this adjustment, the base model's EHC marginal and joint JS rise from 0.0319/0.1001 to 0.1168/0.1850, while the fully covered v3 run stays at 0.0077/0.0363. On EDC, direct SFT's raw 0.0048/0.0160 becomes 0.0377/0.0489 under the same penalty, whereas v3, which returns all 400 rows, is unaffected at 0.0040/0.0215.

We therefore treat v3 as the cleanest headline result: it achieves full coverage on both configurations and the lowest raw JS. Direct SFT still shows a large improvement, but part of its apparent advantage on EDC is due to the 19 missing rows. We report raw and coverage-adjusted values in Table 4 and recommend that future work report both.

### 6.6 Discussion

The FAS results support C2 for EDC: supervised fine-tuning on a reconstructed joint distribution reduces both marginal and joint divergence by roughly an order of magnitude, and the gain is not explained by retrieval or quota sampling. For EHC the gain is real but smaller, and the KL diagnostic moves in the opposite direction for the joint. The honest summary is that the method works well when the underlying temporal trends are stable and the attributes are correlated, and less well when they are not.

This is also the part of the pipeline where the language model contributes something the tabular baseline cannot: it generates a synthetic population rather than a point prediction, and it can be asked to condition, perturb, or correct that population in natural language. The individual-level results in Section 5 are competitive rather than superior; the population-level results are the clearest evidence for the value of the approach.
