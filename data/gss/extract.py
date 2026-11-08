import pyreadstat
import pandas as pd
import sys

# 用法: python dta_to_csv.py input.dta output.csv
# if len(sys.argv) != 3:
#     print("Usage: python dta_to_csv.py input.dta output.csv")
#     sys.exit(1)

input_path = r"D:\School\LLMs\Context engineering\DATA\GSS\GSS2014.dta"
output_path = r"D:\School\LLMs\Context engineering\DATA\GSS\GSS2014.csv"

# 读取 .dta（pyreadstat 会保留标签信息，可通过参数控制）
df, meta = pyreadstat.read_dta(input_path)

# 如果想把带标签的数值转换成字符串类别，可以:
# df = df.apply(lambda col: col.cat.codes if hasattr(col, "cat") else col)

# 保存为 CSV（index=False 通常更合适）
df.to_csv(output_path, index=False, encoding="utf-8")
print(f"Saved CSV to {output_path}")