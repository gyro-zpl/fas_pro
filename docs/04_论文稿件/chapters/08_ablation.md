## 8 Ablations

### 8.1 Setup

We isolate three design choices with matched training runs. **A1 (random sampling)** replaces the nine-strategy pool with a uniform random sample of the same size (6,516 rows) from all cleaned 2016--2020 rows. **A2 (minimal prompt)** keeps the same rows but removes the interaction block from the prompt (party alignment, ideological distance, age difference). **A3 (name-only prompt)** additionally removes the candidate's party, position, and age, leaving only the candidate name. All three use the same LoRA configuration, three epochs, and seed 42; they are evaluated on the same in-distribution rows (2016--2019) as the main model. The full model is reported as the mean of seeds 42--44.

### 8.2 Results

Table 5 reports the results. The full model reaches an MAE of 20.84 $\pm$ 0.24, a correlation of 0.679 $\pm$ 0.003, and a choice accuracy of 0.553 $\pm$ 0.011. A1 reaches 21.27, 0.658, and 0.543; A2 reaches 21.13, 0.670, and 0.567; A3 reaches 21.45, 0.657, and 0.565.

The direction is consistent: removing the distribution-aware sampling (A1) or the candidate metadata (A3) worsens MAE by roughly half a point and lowers the correlation by about 0.02. Removing the interaction block (A2) has a smaller effect on MAE and correlation, although it is still below the full model. Choice accuracy does not separate the conditions reliably; the differences are within the seed-level variation of the full model.

### 8.3 Interpretation

The ablations support a modest version of the data-design claim. The pipeline's sampling and metadata choices contribute a consistent 2--3% improvement in MAE and a small gain in correlation, but they are not the main driver of the result. The interaction features, which are the most theoretically motivated part of the design, have the smallest measured effect in-distribution. This is worth stating plainly: the data-design story is real but small, and the paper should not claim that each component is individually necessary.

The small average effect does not mean the design is irrelevant. The sampling strategies target tails and cross-pressure cases that are rare in a uniform sample, and the cross-pressure analysis below shows where the model's behavior differs from the tabular baseline even when average MAE is similar.

### 8.4 Cross-pressure subgroups

We classify each voter--candidate pair into three conflict types from Section 3.4 and compute MAE and correlation within each type on the 2016--2019 rows (Table 5, lower block). The fine-tuned LLM is substantially better than HistGB on CP3, the identity-versus-party conflict (MAE 15.51--17.80 across seeds versus 20.31 for HistGB), and comparable on CP1, the party-versus-ideology conflict (18.77--21.55 versus 21.55). It is slightly worse on CP2, the ideology-versus-party conflict (23.21--24.13 versus 23.18), and on the non-conflict rows (20.45--21.09 versus 20.14).

This is a finer-grained version of the main result: the LLM does not win on average, but it handles one theoretically important conflict type better than the feature-engineered baseline. The pattern is consistent across the three seeds, although each cell is small (154--318 rows), so we report it as a descriptive subgroup result rather than a confirmed advantage.

### 8.5 Limitations

The ablations use one seed each, and the effects are small; a two- or three-seed replication would be needed to establish that the A1/A3 gaps are larger than run-to-run variation. The choice metric is noisier than MAE and should not be used to rank the ablations. Finally, the cross-pressure cells are small and the CP3 advantage should be treated as a hypothesis for future work rather than a settled finding.
