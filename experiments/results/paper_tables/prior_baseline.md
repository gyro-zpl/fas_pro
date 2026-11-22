# Prior-only FAS baseline

Reconstructed Q* from [2004, 2008, 2012, 2014, 2016, 2018, 2021, 2022]; target 2024; 400 draws; base seed 42; 1,000 bootstrap resamples of the generated rows.

| Config | Prior marginal JS [95% CI] | Prior joint JS [95% CI] | Seed range marginal | Seed range joint |
|---|---|---|---|---|
| EDC | 0.0004 [0.0002, 0.0023] | 0.0076 [0.0070, 0.0141] | 0.0003--0.0009 | 0.0034--0.0076 |
| EHC | 0.0011 [0.0006, 0.0036] | 0.0225 [0.0192, 0.0308] | 0.0009--0.0033 | 0.0150--0.0239 |

## Comparison with the LLM runs

Delta is prior minus LLM; negative means the prior is closer to the empirical 2024 sample. CIs resample each run independently.

| Config | vs | Delta marginal JS [95% CI] | Delta joint JS [95% CI] |
|---|---|---|---|
| EDC | v3 | -0.0036 [-0.0069, -0.0011] | -0.0138 [-0.0210, -0.0067] |
| EDC | sft | -0.0045 [-0.0079, -0.0017] | -0.0084 [-0.0147, -0.0026] |
| EDC | base | -0.0549 [-0.0632, -0.0467] | -0.1105 [-0.1272, -0.0938] |
| EHC | v3 | -0.0066 [-0.0109, -0.0030] | -0.0138 [-0.0232, -0.0059] |
| EHC | sft | -0.0156 [-0.0206, -0.0110] | -0.0245 [-0.0353, -0.0144] |
| EHC | base | -0.0308 [-0.0384, -0.0248] | -0.0776 [-0.0943, -0.0616] |

## Reproduce

```bash
python -m experiments.eval.prior_baseline \
    --data-dir /home/jpzhao_team/cyyuan/FAS_PRO/data/gss --samples 400 --seed 42 --bootstrap 1000
```