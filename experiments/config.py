"""Single source of truth for paths and model registry.

All new experiment code imports from here. Old scripts are left untouched.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(os.environ.get("FAS_ROOT", "/home/jpzhao_team/cyyuan/FAS_PRO"))

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
DATA = ROOT / "data"
ANES_FEATURES = DATA / "anes" / "processed" / "anes_all_years_with_features.csv"
ANES_GROUPS_LONG = DATA / "anes" / "extend_features" / "data" / "anes_all_years_groups_long.csv"
GROUP_METADATA = DATA / "anes" / "extend_features" / "data" / "group_metadata.csv"

# Existing sampled training file. Used as the training pool so we keep the
# prior sampling design instead of training on all ~95k cleaned rows.
TRAIN_SOURCE = DATA / "anes" / "finetune" / "train_dataset_fixed.jsonl"

V3 = DATA / "anes" / "v3"
V3_SPLITS = V3 / "splits"
V3_GROUPS = V3 / "groups"

GSS_DIR = DATA / "gss"
GSS_RESULTS = GSS_DIR / "test_results"

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
# The original experiments used Llama-3.1-8B-Instruct, but its HF weights are
# no longer present on this machine and the network is blocked. Qwen3-8B is
# available locally. The registry keeps both so switching is a one-line change.
LLAMA_LOCAL = Path("/home/jpzhao_team/models/llama-3.1-8b-instruct")

MODELS = {
    "qwen3-8b": {
        "path": "/home/jpzhao_team/models/Qwen--Qwen3-8B/b968826d9c46dd6066d109eabc6255188de91218",
        "family": "qwen",
        "enable_thinking": False,
    },
    "qwen3-8b-base": {
        "path": "/home/jpzhao_team/models/Qwen3-8B-Base",
        "family": "qwen",
        "enable_thinking": False,
    },
    "llama-3.1-8b": {
        # Uses the local copy once `experiments/download_llama.sh` has run.
        "path": str(LLAMA_LOCAL) if LLAMA_LOCAL.exists() else "meta-llama/Llama-3.1-8B-Instruct",
        "family": "llama",
        "enable_thinking": None,
    },
    "deepseek-r1-distill-llama-8b": {
        "path": "/home/jpzhao_team/models/deepseek-r1-distill-llama-8b",
        "family": "deepseek",
        "enable_thinking": None,
    },
}

DEFAULT_MODEL = os.environ.get("FAS_MODEL", "qwen3-8b")

# LoRA adapters produced by the new pipeline (paths are placeholders until D5).
LORA_DIR = ROOT / "sft" / "sft_models"
LORA_CANDIDATE = LORA_DIR / "anes_qwen3_8b_lora_v3"
LORA_GROUP = LORA_DIR / "anes_groups_qwen3_8b_lora_v3"

# Legacy Llama adapters (cannot be loaded without Llama weights).
LEGACY_LORA_V2 = LORA_DIR / "anes_llama3_lora_v2" / "final_model"

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------
SEEDS = [42, 43, 44]
RESULTS = ROOT / "experiments" / "results"
PREDS = RESULTS / "preds"
for _p in (RESULTS, PREDS):
    _p.mkdir(parents=True, exist_ok=True)

# Candidate years
TRAIN_YEARS = [2016, 2018, 2019, 2020]
VAL_YEARS = [2022]
TEST_YEARS = [2024]

# Clean candidate names: drop social-group targets that were mixed into the
# candidate table, and drop measurements of the same person/candidate pairs.
GROUP_LIKE_TARGETS = {"black", "white", "dem", "rep"}

# Prompt level: minimal / core / full (matches generate_finetune_data.py)
PROMPT_LEVEL = "core"

# Evaluation
RELAXED_BIN = 20
RELAXED_TOL = 10


def model_info(key: str | None = None) -> dict:
    key = key or DEFAULT_MODEL
    if key not in MODELS:
        raise KeyError(f"Unknown model {key!r}. Available: {list(MODELS)}")
    return {"key": key, **MODELS[key]}
