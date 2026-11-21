# Second model family: DeepSeek-R1-Distill-Llama-8B

One seed (42), same `train_t` data and LoRA recipe as Qwen3-8B.
The model was trained to emit `</think>\n<score>`; parse rate 100%.

| Model | 2016–2019 MAE | corr | choice | 2024 MAE | corr | choice |
|---|---|---|---|---|---|---|
| Qwen3-8B (3 seeds) | 20.84 | 0.679 | 0.518 | 36.47 | −0.029 | 0.527 |
| DeepSeek-R1-Distill-Llama-8B (1 seed) | 22.66 | 0.635 | 0.548 | 41.44 | −0.017 | 0.523 |
| HistGB | 20.58 | 0.714 | 0.534 | 35.06 | −0.030 | 0.456 |

Sources:
- `experiments/results/preds/deepseek_test_r_generate.predictions.jsonl`
- `experiments/results/preds/deepseek_test_t_generate.predictions.jsonl`

Interpretation: the qualitative pattern (in-distribution signal, 2024 collapse)
reproduces across families; Qwen has lower MAE/higher correlation, DeepSeek has
slightly higher choice. With one seed, differences are descriptive.
