#!/bin/bash
# 配额退火系统运行脚本（简洁版）
# 模仿RAG.sh的风格

# 设置API配置
export DIFY_API_URL='http://10.113.178.150:2001/v1/chat-messages'
export DIFY_API_KEY='app-TFkxG9dj0YC5Fj0jUv11XctG'

# ============================================
# 任务1: ehc组合（环境、健康、城市）
# ============================================
export COLUMNS='["natenvir", "natheal", "natcity"]'
python /home/cyyuan/FAS_PRO/codes/GSS/quota_annealing_runner.py \
    --columns natenvir natheal natcity \
    --mission "ehc" \
    --quota_configs /home/cyyuan/FAS_PRO/codes/GSS/configs/ehc_quotas.json \
    --target_total 400 \
    --batch_size 50 \
    --max_iterations 15 \
    --output_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_ehc.csv \
    --output_report /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_ehc_report.json

# ============================================
# 任务2: edc组合（教育、毒品、犯罪）
# ============================================
export COLUMNS='["nateduc", "natdrug", "natcrime"]'
python /home/cyyuan/FAS_PRO/codes/GSS/quota_annealing_runner.py \
    --columns nateduc natdrug natcrime \
    --mission "edc" \
    --quota_configs /home/cyyuan/FAS_PRO/codes/GSS/configs/edc_quotas.json \
    --target_total 400 \
    --batch_size 50 \
    --max_iterations 15 \
    --output_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_edc.csv \
    --output_report /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_edc_report.json

# ============================================
# 任务3: 单配额示例（natcity='3'占20%）
# ============================================
export COLUMNS='["natcity", "natenvir", "natheal"]'
python /home/cyyuan/FAS_PRO/codes/GSS/quota_annealing_runner.py \
    --columns natcity natenvir natheal \
    --mission "Generate samples with natcity=3 at 20%" \
    --quota_type single \
    --quota_attr natcity \
    --quota_value 3 \
    --quota_target 0.20 \
    --quota_tolerance 0.05 \
    --quota_priority 4 \
    --target_total 300 \
    --batch_size 50 \
    --max_iterations 12 \
    --output_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_quota_single.csv

echo ""
echo "=========================================="
echo "全部任务完成！"
echo "=========================================="

