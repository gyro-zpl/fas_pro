export DIFY_API_URL='http://10.113.178.150:2001/v1/chat-messages'
export DIFY_API_KEY='app-TFkxG9dj0YC5Fj0jUv11XctG'

export COLUMNS='["natenvir", "natheal", "natcity"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_RAG.py --mission ehc --batch_size 20 --epochs 20 --out_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_ehc_v2.csv --out_raw_path None

export COLUMNS='["nateduc", "natdrug", "natcrime"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_RAG.py --mission edc --batch_size 20 --epochs 20 --out_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_edc_v2.csv --out_raw_path None

export COLUMNS='["c1", "c2", "c3"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_RAG.py --mission "natrace:Improving the conditions of Blacks; natarms:The military, armaments, and defense; nataid:Foreign aid" \
 --batch_size 20 \
 --epochs 20 \
 --out_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_RAG_sft_fanhua_v2.csv \
 --out_raw_path None