# experiments/

Self-contained layer for the 20-day rebuild. Old scripts under `codes/` and `sft/`
are left in place; this directory reads their outputs and writes new artifacts.

## Layout

```
experiments/
├── config.py              # paths + model registry (single source of truth)
├── smoke_test.py          # load model, generate 3 ratings
├── data/
│   ├── audit_raw.py       # raw-table audit -> results/00_audit.md
│   ├── clean_candidates.py# normalize names, drop groups, dedupe
│   ├── build_splits.py    # temporal + respondent splits
│   ├── build_group_splits.py
│   └── prompts.py         # candidate/group prompt builders
├── eval/
│   ├── metrics.py         # MAE/RMSE/relaxed ACC/choice/pairwise/ECE
│   ├── readout.py         # model loading, chat template, number parsing
│   └── evaluate.py        # CLI evaluation
├── baselines/
│   └── tabular.py         # HistGB + Ridge on the same features
├── tests/                 # plain-assert tests (pytest not installed)
├── results/               # reports, predictions, logs
└── run_baselines.sh       # background runner
```

## Environment

Use the `llama_factory` conda env (it is the only env under the project account with
torch + transformers + peft + datasets + sklearn):

```bash
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python
```

Note: there is **no** conda env named `FAS_PRO` on this machine. The existing envs are
`FAS` (missing `peft`), `langchain`, `langchain1.0`, `llama_factory`, `TSA-Net`, plus
`harry/sft` and `harry/verl`. The `conda` launcher itself has a stale
`/home/cyyuan` shebang, so use absolute python paths.

## Run order

```bash
cd /home/jpzhao_team/cyyuan/FAS_PRO

# 1. audit
$PY -m experiments.data.audit_raw

# 2. clean
$PY -m experiments.data.clean_candidates

# 3. splits
$PY -m experiments.data.build_splits
$PY -m experiments.data.build_group_splits

# 4. tests
$PY -m experiments.tests.test_metrics
$PY -m experiments.tests.test_readout

# 5. tabular baselines
OMP_NUM_THREADS=4 $PY -m experiments.baselines.tabular

# 6. LLM smoke test (GPU 3)
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 $PY -m experiments.smoke_test

# 7. full baseline eval suite (background)
setsid nohup bash experiments/run_baselines.sh \
  > experiments/results/logs/run_baselines.out 2>&1 < /dev/null &
```

## R protocol (main PAS table)

The thesis main table uses the respondent-held-out-number R protocol, not the
temporal-protocol T models. The R artifacts are committed for reproduction:

```bash
cd /path/to/fas_pro_push   # repository root; config.ROOT defaults here

# tabular baselines trained on train_r
$PY -m experiments.baselines.tabular --train-split train_r --suffix _r

# Qwen3-8B, 3 seeds, parallel on GPUs 0/1/3
bash experiments/run_train_r_parallel.sh

# 2024 test_t / strict T x R (seed44 was interrupted and retrained)
bash experiments/run_seed44_r.sh 1

# DeepSeek-R1-Distill-Llama-8B, one seed
bash experiments/run_deepseek_r.sh 2

# evaluate: seed42/test_r/test_t, then seed43, seed44
bash experiments/run_eval_r_suite.sh 0 42:test_r:generate 42:test_t:generate
bash experiments/run_eval_r_suite.sh 1 43:test_r:generate 43:test_t:generate
bash experiments/run_eval_r_suite.sh 3 44:test_r:generate 44:test_t:generate

# tables + paired/cluster bootstrap + cross-pressure
$PY -m experiments.eval.r_protocol_tables --seeds 42,43,44
```

Artifacts:

| File | Content |
|---|---|
| `results/paper_tables/r_protocol.md` / `.json` | main R table, CIs, paired bootstrap |
| `results/paper_tables/r_pas_by_year.md` | R protocol per-year |
| `results/paper_tables/r_cross_pressure.md` | R cross-pressure subgroups |
| `results/paper_tables/r_test_t.md` | 2024 full test set |
| `results/paper_tables/r_test_tr.md` | strict 226-row T×R cell |
| `results/01_tabular_r.md` / `.json` | HistGB/Ridge R baselines |
| `results/preds/qwen3_r_*` / `deepseek_r_*` | committed predictions and summaries |
| `results/logs/train_r_*` / `eval_r_*` | committed R training/eval logs |
| `experiments/runs.csv` r030–r043 | run registry |

The 87 MB LoRA adapters themselves are excluded by `.gitignore`; their configs
and SHA256 checksums are in `sft/sft_models/README.md`. The committed
predictions are sufficient to regenerate every R table without loading the
weights.

## Person-linkage audit

The v4 person-level rerun plan requires a cross-wave person identifier. The
released ANES CSVs do not carry one: `caseid` is a within-wave 1..N number
(cross-wave birth-year agreement about 2%), the 2020 file only has a random
`ResponseId`, and the clean table's `respondent_id` is 0..N-1. The R protocol
is therefore a held-out-number protocol, not a person-level split. The audit
is reproducible with:

```bash
python -m experiments.data.audit_person_linkage \
  --raw-dir /path/to/FAS_PRO/data/anes/raw_data \
  --anes-root /path/to/FAS_PRO/data/anes/v3
```

Output: `results/00_person_linkage_audit.md` / `.json`. See
`docs/03_进度与执行/人物级链接审计_2026-11-20.md` for the interpretation.

## Covariate audit and dropout probe

The per-year analysis is sensitive to what each wave actually measures. The
audit reports missingness, the number of political profiles the prompt can
render, and coding direction:

```bash
python -m experiments.data.audit_covariates
```

Output: `results/00_covariate_audit.md` / `.json`. In 2024 all 3,421 prompts
render one political profile; 2020 has no ideology item and a reversed `pid7`;
2018 stores `pid7` as a sentinel.

The dropout probe asks whether that input loss explains the 2024 collapse. It
reuses the committed seed-42 2019 predictions, runs the same LoRA adapter on the
same rows with the political block replaced by the 2024 defaults, and evaluates
HistGB on the original and ablated features:

```bash
CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 python -m \
  experiments.diagnostics.covariate_dropout_probe \
  --lora /path/to/qwen3_8b_r_seed42
```

Output: `results/diagnostics/covariate_dropout_probe.md` / `.json` and the
ablated predictions. On 1,302 2019 rows, LLM correlation falls from 0.714 to
0.166 and HistGB from 0.734 to 0.172; MAE rises from 19.34 to 31.67 and 19.04 to
31.46. Most of the 2024 loss is input loss, with a smaller residual shift.

## Model registry

`config.MODELS` currently contains:

| key | path | note |
|---|---|---|
| `qwen3-8b` | local HF cache | **default**; Llama weights are absent on this machine |
| `qwen3-8b-base` | local base model | for base-vs-instruct checks |
| `llama-3.1-8b` | HF id | requires downloading weights |

The original Llama LoRA adapters (`sft/sft_models/anes_llama3_lora_v2/final_model`)
cannot be loaded without the Llama base weights. They are kept for reference only.

## Splits

| file | rows | meaning |
|---|---|---|
| `train_t.jsonl` | 6,516 | sampled training pool, 2016-2020 |
| `val_t.jsonl` | 2,501 | 2022 (capped by respondents) |
| `test_t.jsonl` | 3,421 | 2024 (all) |
| `train_r.jsonl` | 5,094 | 80% respondents, 2016-2020 |
| `test_r.jsonl` | 6,014 | 20% respondents, all years |
| `test_tr.jsonl` | 226 | 2024 ∩ test respondents |

Group splits:

| file | rows | groups |
|---|---|---|
| `train_2020.jsonl` / `test_2020.jsonl` | 27,423 total | 9 (2020 block split) |
| `train_r.jsonl` / `test_r.jsonl` | 34,287 total | 6 (multi-year respondent split) |

## Known data issues (documented, not hidden)

1. **Candidate table pollution**: 8,165 trailing-dot names (`obama.`, `hrc.`) and
   23,060 social-group rows (`black/white/dem/rep`) were removed.
2. **Panel structure (number-level, not person-level)**: the released files do not
   carry a cross-wave person key; `respondent_id` is a within-wave number. Counts
   such as 3,080 / 3,165 and "zero test-only respondents" are number overlaps, not
   panel evidence. Temporal splits therefore reuse respondents; the R protocol
   holds out respondent numbers within a wave, not linked persons. See the
   person-linkage audit above.
3. **2020 group rows lost `respondent_id`**: block-level pseudo IDs are assigned by
   detecting repeated group names. 2020 is excluded from respondent-level group
   splits but included in the 2020 block split.
4. **Group coverage is uneven by year** (2016: 3, 2018: 6, 2019: 4, 2020: 9, 2022: 2),
   so group temporal splitting is not meaningful.
5. **Prompt collisions**: 715 prompts have conflicting targets because different
   respondents can share every visible attribute. This is irreducible label noise,
   not leakage.

## Outputs

- `results/00_audit.md` / `.json`
- `results/01_tabular.md` / `.json`
- `results/preds/*.predictions.jsonl` and `*.summary.json`
- `results/logs/*.log`
