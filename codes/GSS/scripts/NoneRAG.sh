
export DIFY_API_URL='http://10.113.178.150:2001/v1/chat-messages'
export DIFY_API_KEY='app-GhDluzpqgOPjYS8XXJDER0n9'
export COLUMNS='["natenvir", "natheal", "natcity"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_none_RAG.py --mission FAS --variables natenvir,natheal,natcity --batch_size 20 --epochs 20 --out_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_none_RAG_sft_ehc_v3.csv --out_raw_path None

export COLUMNS='["nateduc", "natdrug", "natcrime"]'
python /home/cyyuan/FAS_PRO/codes/GSS/DIFY_none_RAG.py --mission FAS --variables nateduc,natdrug,natcrime --batch_size 20 --epochs 20 --out_csv /home/cyyuan/FAS_PRO/data/gss/test_results/gss_31_none_RAG_sft_edc_v3.csv --out_raw_path None