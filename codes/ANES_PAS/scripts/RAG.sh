export DIFY_API_URL='http://10.113.178.150:2001/v1/chat-messages'
export DIFY_API_KEY='app-GhDluzpqgOPjYS8XXJDER0n9'
export CUDA_VISIBLE_DEVICES=1
python /home/cyyuan/FAS_PRO/codes/ANES_PAS/DIFY_PAS.py \
    --mission "Donald Trump" \
    --input_path /home/cyyuan/ACL2025/Data/Anes2020/selected_anes2020.csv \
    --out_path /home/cyyuan/FAS_PRO/codes/ANES_PAS/correct.txt \

