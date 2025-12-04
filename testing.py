import pandas as pd
import argparse

# 1. Set up CLI argument parser
parser = argparse.ArgumentParser(description="Read a Parquet file and print the 'response' column.")
parser.add_argument(
    "file_name", 
    nargs="?",  # makes it optional
    default="./data/output/model_llama/user_Test/part_0000_20251125-141912.parquet",
    help="Path to the Parquet file (default: %(default)s)"
)
args = parser.parse_args()

# 2. Load the file
df = pd.read_parquet(args.file_name)

# 3. Print each response
for i, (index, row) in enumerate(df.iterrows()):
    print(i, row)
