#!/bin/bash
# Download Llama-3.1-8B-Instruct.
# Prerequisite: log in as the token's account and click "Agree and access repository" at
#   https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
#
# Usage:
#   export HF_TOKEN=hf_xxx
#   bash experiments/download_llama.sh
set -euo pipefail

ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
PY=/home/jpzhao_team/cyyuan/miniconda3/envs/llama_factory/bin/python

# The local proxy (127.0.0.1:17898) is dead; direct access works.
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY || true

if [ -z "${HF_TOKEN:-}" ] && [ -f /home/jpzhao_team/.cache/huggingface/token ]; then
  export HF_TOKEN="$(cat /home/jpzhao_team/.cache/huggingface/token)"
fi
: "${HF_TOKEN:?export HF_TOKEN=hf_xxx first}"

cd "$ROOT"
"$PY" -u -m experiments.download_llama
