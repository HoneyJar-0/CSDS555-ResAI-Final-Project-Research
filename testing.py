import pandas as pd
file_name = "./data/output/model_llama/user_Test/part_0000_20251125-141912.parquet"
df = pd.read_parquet(file_name)
for i, r in enumerate(df["response"]):
    print(i, "\n", r.strip(), '\n')
