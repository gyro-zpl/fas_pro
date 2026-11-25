## 4 Method

### 4.1 Task definitions

We study two levels of survey simulation. Figure 1 gives an overview of the two pipelines and the three evaluation levels.

**Partial Attribute Simulation (PAS).** Given a voter profile $v$, a target $c$ with metadata $\mathcal{M}[c]$, and a year $t$, the model predicts a feeling-thermometer score

$$
\hat y = f_\theta(v,\mathcal{M}[c],t)\in[0,100].
$$

For evaluation we also form a discrete choice. For each respondent-year pair we collect the candidates the respondent rated, take the model's highest-scoring candidate

$$
\hat c^{\ast} = \arg\max_{c\in\mathcal{C}(v,t)} f_\theta(v,\mathcal{M}[c],t),
$$

and compare it with the candidate with the highest ground-truth rating. We call this **rating-induced choice**; it avoids requiring a separate vote-choice variable, which is unavailable for all waves.

**Full Attribute Simulation (FAS).** Given three categorical policy attitudes $(A,B,C)$, the model must generate samples whose marginal and joint distributions match a target population. We evaluate the generated sample against the empirical 2024 distribution.

### 4.2 Prompting and LoRA fine-tuning

Each PAS example is rendered as a short survey vignette. The system message defines the task and fixes the output format (a single integer in $[0,100]$). The user message contains the voter profile (age, gender, race, education, party identification, ideology), the candidate block (name, party, position, age), and the interaction block (party alignment, ideological distance, age difference, year). Group examples use the same structure with a group block instead of a candidate block.

We fine-tune a compact instruction-tuned model, Qwen3-8B, introduced by Yang et al. (2025), with LoRA (Hu et al., 2022). For a frozen weight matrix $W_0$, LoRA learns a low-rank update

$$
W' = W_0 + \frac{\alpha}{r}BA,
\qquad B\in\mathbb{R}^{d\times r},\ A\in\mathbb{R}^{r\times k},
$$

with rank $r=8$, scaling $\alpha=32$, dropout $0.1$, applied to the query, key, value, output, gate, up, and down projections. We train for three epochs with a learning rate of $10^{-4}$, an effective batch size of 16, and a maximum sequence length of 512. The loss is the causal language-modeling loss on the answer tokens only; prompt tokens are masked with $-100$. This yields 21.8M trainable parameters, or 0.27% of the model. We fine-tune three seeds (42, 43, 44) and report mean and standard deviation.

### 4.3 Rating readouts and choice

A fine-tuned model can be read out in two ways.

**Generation.** We decode greedily for at most eight tokens and parse the first integer in $[0,100]$. This preserves individual variation and is our primary readout for choice.

**Anchor likelihood.** Because the tokenizer splits most two- and three-digit numbers into several tokens, a single-position softmax over 0--100 is not available. We instead score eleven anchor strings $a\in\{0,10,\dots,100\}$ by their sequence log-probability under the prompt,

$$
s(a)=\sum_{k=1}^{|a|}\log p_\theta\!\left(a_k \mid \text{prompt}, a_{<k}\right),
$$

normalize with a softmax over the eleven anchors, and report both the expectation (a calibrated scalar) and the argmax (a discrete rating). The anchor readout requires one forward pass per anchor but no generation, and it exposes a full distribution over the rating scale.

For choice we use the same rule in both readouts: within each respondent-year group, the candidate with the highest predicted score is selected. We report the observed accuracy together with the chance accuracy $1/|\mathcal{C}(v,t)|$, which is 0.5 for the two-candidate 2024 wave. Because the model emits integer ratings, ties are common; we therefore also report **tie-aware choice**, which splits credit across tied candidates instead of relying on row order.

### 4.4 Baselines

We compare against three baselines. **HistGB** (Friedman, 2001) and **Ridge** are trained on the same tabular features used to render the prompts (demographics, party, ideology, candidate metadata, and the interaction features of Section 3.4); they provide a strong non-LLM reference and test whether the language model adds anything beyond feature engineering. The **base LLM** uses the same prompts without fine-tuning. All baselines are evaluated on the same splits and metrics.

### 4.5 Population-level simulation (FAS)

FAS has three stages.

**Extrapolation.** We observe a short time series for the three marginals and one pairwise table (B--C) and predict the target year with a four-model ensemble,

$$
\hat p(t^{\ast}) =
\frac{1.0\,\hat p_{\text{poly}}+1.0\,\hat p_{\text{loess}}+0.6\,\hat p_{\text{exp}}+0.8\,\hat p_{\text{lin}}}{3.4},
$$

where the components are quadratic regression, LOESS with a tri-cube kernel, an exponential trend, and linear regression. Predictions are clipped to $[0,1]$ and renormalized.

**Joint reconstruction.** The predicted marginals and the B--C table are combined with iterative proportional fitting (IPF). Starting from the independence approximation $Q^{(0)}(a,b,c)=P(A)P(B)P(C)$, we alternate multiplicative updates to match the marginals and the B--C table, run 300 iterations, normalize, and apply Laplace smoothing ($\alpha=10^{-6}$). The other two pairwise joints (A--B, A--C) are read out from the resulting joint. The output is a coherent three-way distribution $Q^{\ast}(A,B,C)$.

**Task construction.** We sample synthetic populations from $Q^{\ast}$ and convert them into five instruction-following task types: batch generation (50%), prior-constrained sampling (25%), scenario variation (15%), subgroup analysis with Wilson confidence intervals (7%), and iterative correction (3%). A compact model is fine-tuned on this mixture. At evaluation time we compare generated samples with the empirical 2024 distribution.

FAS samples were generated with 8B Llama-family models (base `llama3:8b`; GSS fine-tuned `llama-gss-v3:8B`) through a local orchestration layer, at 20 rows per batch for 20 batches; failed batches were skipped, giving 351--400 valid rows.

### 4.6 Metrics and reproducibility

For PAS we report mean absolute error (MAE), root mean squared error (RMSE), Pearson and Spearman correlation, a relaxed accuracy (same 20-point bin or absolute error $\le 10$), rating-induced choice accuracy, and expected calibration error (ECE). For FAS we report Jensen--Shannon divergence (Lin, 1991), abbreviated JS, and symmetric KL divergence on each marginal and on each pairwise joint, averaged over the three attributes or three pairs.

**Analysis plan.** Two quantities are pre-specified. The first is the share of the base-to-HistGB gap that fine-tuning closes, reported with paired bootstrap intervals. The second is the middle-minus-tail contrast in the error profile---the mean MAE difference between the two middle true-score quartiles and the two outer ones---tested with a paired permutation test over respondents and a respondent-cluster bootstrap, with Holm correction across the 12 year-by-quartile cells. The second is the paper's primary inferential test.

All results are produced by scripts under `experiments/`; predictions are saved per row, and every table in the paper is generated from those files. The main PAS results use three seeds; the ablations and the temporal mitigation use one seed and are reported as such. The R-protocol predictions and paper tables are committed to the repository; the 87 MB LoRA adapters are excluded and can be retrained with the scripts in `experiments/` (see `sft/sft_models/README.md`).
