#!/bin/bash
# One-screen status for the 20-day rebuild.
# Usage: bash experiments/status.sh
set -u
ROOT=/home/jpzhao_team/cyyuan/FAS_PRO
cd "$ROOT" || exit 1

echo "========== $(date '+%F %T') =========="

echo
echo "--- running ---"
ps -eo pid,etime,pcpu,cmd | grep -E "train_lora|eval\.evaluate" | grep -v grep \
  || echo "(no training/eval process)"

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu \
  --format=csv,noheader

echo
echo "--- training log (last 4 lines) ---"
for f in experiments/results/logs/train_resume.log \
         experiments/results/logs/train_seeds_current.log; do
  [ -f "$f" ] || continue
  echo "[$f]"
  tail -2 "$f" 2>/dev/null | tr '\r' '\n' | tail -4
done

echo
echo "--- checkpoints ---"
for d in sft/sft_models/qwen3_8b_t_seed*; do
  [ -d "$d" ] || continue
  latest=$(ls -d "$d"/checkpoint-* 2>/dev/null | sort -V | tail -1)
  final=""
  [ -f "$d/adapter_model.safetensors" ] && final=" + final adapter"
  echo "  $(basename "$d"): ${latest:-no checkpoint}${final}"
done

echo
echo "--- latest result summaries ---"
ls -lt experiments/results/preds/*.summary.json 2>/dev/null | head -5 \
  || echo "(none yet)"

echo
echo "--- docs ---"
echo "  docs/README.md                              (index + progress dashboard)"
echo "  docs/03_进度与执行/实验进度时间线.md        (timestamped log)"
echo "  docs/03_进度与执行/运行状态_2026-11-05.md   (current status)"
echo "  experiments/results/        (audit, baselines, preds, logs)"
