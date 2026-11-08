#!/bin/bash
# 配额退火系统 - 使用脚本
# 
# 使用方法：bash scripts/quota.sh
# 或者单独运行：python DIFY_with_quota_annealing.py --mode quick

export DIFY_API_URL='http://10.113.178.150:2001/v1/chat-messages'
export DIFY_API_KEY='app-7GfwGeS41LqdDwaYfgVq1geq'

# 任务1: ehc组合 - 环境、健康、城市
export COLUMNS='["natenvir", "natheal", "natcity"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_with_quota_annealing.py ehc \
    --config /home/cyyuan/FAS_PRO/codes/GSS/configs/ehc_quotas.json \
    --target_total 400 --batch_size 50 --max_iterations 15 \
    --output_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_ehc.csv

# 任务2: edc组合 - 教育、毒品、犯罪  
export COLUMNS='["nateduc", "natdrug", "natcrime"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_with_quota_annealing.py edc \
    --config /home/cyyuan/FAS_PRO/codes/GSS/configs/edc_quotas.json \
    --target_total 400 --batch_size 50 --max_iterations 15 \
    --output_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_edc.csv

