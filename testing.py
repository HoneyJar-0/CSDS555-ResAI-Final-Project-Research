import pandas as pd
file_name = "./data/output/model_llama/user_Jacob/part_0000_20251124-213030.parquet"
df = pd.read_parquet(file_name)
for r in df["response"]:
    print(r)
    print('\n')
