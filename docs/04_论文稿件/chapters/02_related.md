## 2 Related Work

### 2.1 LLMs as virtual survey respondents

A growing line of work uses LLMs to simulate survey responses. Argyle et al. (2022) condition language models on demographic backstories and show that the resulting "silicon samples" can approximate subgroup opinions; Santurkar et al. (2023) show that the same models are skewed toward WEIRD populations; Dillion et al. (2023) ask whether models can replace human participants; and Kim and Lee (2023) use models to augment surveys. Later work extends the paradigm to random silicon sampling (Sun et al., 2024), cross-survey probing (Zhou et al., 2024), and the construction of multi-domain benchmarks (Zhao et al., 2025). Cao et al. (2025) specialize models to survey response distributions for global populations.

Most of this work evaluates a model as a one-shot respondent: a profile and a question are placed in the prompt, and the answer is compared with a survey margin. The model is not trained on the survey structure, and the evaluation rarely separates individual-level accuracy from population-level distributional alignment. Our work builds on the benchmark of Zhao et al. (2025) but turns it into a training and evaluation pipeline: we reconstruct interaction-level data, fine-tune a compact model, and report individual, population, and temporal results separately.

### 2.2 Persona, role-play, and personalization

A related literature studies how well models maintain a persona. Park et al. (2024) simulate 1,000 people with generative agents; Wang et al. (2024) evaluate personality fidelity in role-playing; Xu et al. (2024) ask whether models can make persona-driven decisions; Salemi et al. (2024) incorporate user profiles into dialogue; and Ng et al. (2024) measure how well models "echo" a user. This work is primarily about behavioral consistency within a conversation. Our setting is different: the model is not asked to stay in character over a dialogue but to reproduce the statistical structure of survey responses across a population. Persona fidelity and distributional fidelity are complementary, and our data-design choices (candidate metadata, interaction features) can be read as a way to make the persona operational.

### 2.3 Fine-tuning for social simulation

Fine-tuning has been used to align models with political or social preferences (Chu et al., 2023; Zhao et al., 2023), often with the goal of mimicking a group's voice. Our goal is not preference alignment but competence alignment: we want the model to learn the rules that map a voter profile and a candidate to a rating, not to adopt a particular ideology. This distinction matters for evaluation. A model that has learned the rules should generalize to unseen respondents and candidates; a model that has learned a group's voice may reproduce the margin without the mechanism. The respondent-disjoint protocol in Section 3.6 is designed to test the former.

### 2.4 Population-level distribution alignment

Synthetic population generation has a long history outside LLMs. Iterative proportional fitting (Deming and Stephan, 1940; Fienberg, 1970) reconstructs a joint distribution from marginal and lower-order constraints, and LOESS (Cleveland, 1979) provides a flexible univariate smoother. We combine these classical tools with a language model: the joint distribution is reconstructed with an ensemble of extrapolators and IPF, and the model is fine-tuned on samples from the reconstructed joint. The resulting system can generate a population rather than a point prediction, which is the capability that distinguishes FAS from PAS.

### 2.5 Temporal generalization and distribution shift

Survey attitudes change across waves, and a model trained on historical data may not transfer to a future election. Prior work on LLM survey simulation has generally evaluated on a single wave or on a pooled sample, which can hide temporal decay. Our temporal analysis is, to our knowledge, the first to report a monotonic per-year decay of individual-level correlation for both a fine-tuned LLM and a feature-engineered baseline on the same ANES panel. The result is a caution for the use of LLM simulations in forecasting settings and a motivation for treating temporal shift as a separate evaluation axis.

### 2.6 Positioning

Our work sits at the intersection of the three lines above. Unlike prompting-only silicon sampling, we train on interaction-level data. Unlike preference alignment, we target distributional and individual-level fidelity. Unlike prior FAS work, we add a panel-aware evaluation protocol and a temporal boundary analysis. The negative results are part of the contribution: the fine-tuned model is competitive rather than superior at the individual level, and both the LLM and the tabular baseline fail under a four-year shift. We report these findings explicitly so that subsequent work can build on a calibrated picture of what structured alignment does and does not buy.
