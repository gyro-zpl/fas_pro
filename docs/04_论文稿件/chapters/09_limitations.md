## 9 Limitations and Ethical Considerations

**Model coverage.** All fine-tuning experiments use Qwen3-8B. We were unable to evaluate a second model family: the Llama-3.1-8B weights are gated and the account used for this project was not authorized. We therefore cannot claim that the results are model-agnostic. The pipeline itself is model-agnostic---the training and readout code accepts any instruction-tuned causal LM---but the empirical claims are conditional on Qwen3-8B.

**Survey coverage.** The individual-level experiments use ANES and the population-level experiments use GSS, both from the United States. We do not test cross-national surveys, non-English questionnaires, or non-political attitude domains. The data-construction choices (candidate metadata, party and ideology features) are tailored to the ANES schema.

**Group attitudes.** We constructed a nine-group voter--group corpus and two evaluation protocols, but the group model was not evaluated in this paper. Group coverage is uneven across waves (three groups in 2016, nine in 2020, two in 2022), and the 2020 block lacks respondent identifiers. We therefore restrict the claims to candidate-level PAS and GSS FAS and leave the group analysis to future work.

**Panel structure.** ANES is a panel: 3,080 of 3,165 respondents appear in more than one wave, and there are no test-only respondents in 2022/2024. Our respondent protocol is disjoint by respondent, and our temporal protocol is disjoint by year, but the T$\times$R cell is small (226 rows). No protocol in this paper isolates a genuinely new respondent observed only in a future wave, because such respondents do not exist in the data.

**Choice metric.** Choice accuracy is rating-induced: we compare the model's highest-scoring candidate with the candidate with the highest ground-truth thermometer score, not with an observed vote. This avoids requiring a vote-choice variable for every wave, but it measures consistency with ratings rather than with behavior.

**Ablations.** The three ablations use one seed each. The effects are small (0.3--0.6 MAE), and a multi-seed replication would be needed to establish that they exceed run-to-run variation. The cross-pressure subgroup cells are also small (154--318 rows), and the CP3 advantage of the LLM should be treated as descriptive.

**Population-level results.** The FAS evaluation covers two three-attribute configurations. EDC improves strongly; EHC improves less, and the symmetric KL of the joint distribution moves in the opposite direction for direct SFT. The generated samples range from 351 to 400 valid rows, and we report both raw and coverage-adjusted metrics. The temporal extrapolation of the joint distribution is validated only for 2024; we do not test a longer horizon.

**Temporal mitigation.** We test one mitigation, adding the 2022 wave to training, and it does not help. We do not test domain adaptation, continual learning, uncertainty-aware decoding, or retrieval of recent survey statistics. The 2024 collapse is documented, not solved.

**Readout sensitivity.** Generation and anchor likelihood give different pictures. The anchor readout can produce a low ECE when predictions are nearly constant, and we therefore report correlation and prediction spread alongside ECE. The choice of readout should be treated as part of the method, not as a post-hoc detail.

**Ethical considerations.** Synthetic respondents are not a substitute for real survey participants. A model trained on historical data can reproduce and amplify the biases of that data, and its outputs can be mistaken for evidence about a population. The temporal results in this paper are a direct warning: a simulator that looks accurate in-distribution can lose all individual signal four years later. We recommend that any deployment report the training era, the validation protocol, the model family, and the coverage of the generated sample, and that synthetic results be clearly labeled as such. The ANES and GSS data are public and were used under their respective terms; no personally identifying information is released with this work.
