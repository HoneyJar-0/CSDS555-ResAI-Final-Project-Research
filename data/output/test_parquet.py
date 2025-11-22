import pandas as pd
import glob
import pyarrow as pa
import pyarrow.parquet as pq

BASE_DIR = "./data/output/dataset"

paths = glob.glob(f"{BASE_DIR}/**/*.parquet", recursive=True)

print(f"Found {len(paths)} parquet files")

dfs = []
common_fields = set()

# First pass — discover ALL fields across files
for p in paths:
    schema = pq.read_schema(p)
    common_fields |= set(schema.names)

common_fields = sorted(common_fields)
print("Unified schema:", common_fields)

# Second pass — load & align schemas
for p in paths:
    df = pd.read_parquet(p)

    # Add missing columns as null to match unified schema
    for col in common_fields:
        if col not in df.columns:
            df[col] = None

    df = df[common_fields]   # reorder columns
    dfs.append(df)

# Final merged DF
df_all = pd.concat(dfs, ignore_index=True)
df.groupby("UUID")

print(df_all.head())
print(df_all.shape)
