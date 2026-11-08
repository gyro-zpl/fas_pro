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
2. **Panel structure**: 3,080 / 3,165 respondents appear in multiple years; there
   are **zero** test-only respondents. Temporal splits therefore reuse respondents.
   This is why we also report respondent-disjoint splits.
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
