import pandas as pd
import pyreadstat

# 读取 .dta（支持 Stata 8-15+）
df, meta = pyreadstat.read_dta("/home/cyyuan/FAS_PRO/data/gss/GSS2024.dta")


# 保存为 CSV（默认 UTF-8）
df.to_csv("/home/cyyuan/FAS_PRO/data/gss/GSS2024.csv", index=False)