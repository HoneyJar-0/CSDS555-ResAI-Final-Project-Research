import pandas as pd
from pprint import pprint
import sys

#sys.path.append('./data/input')

from parquet_creation import generate_prompt


def test_read_parquet():
    print("reading file...")
    df = pd.read_parquet(
        "./data/input/dataset.parquet",
        filters=[("UUID", ">", 1_000_000), ("UUID", "<", 1_500_000)]
    )

    print("parquet head:")
    pprint(df.head(20))

    print("COLUMNS:", df.columns.tolist())


    print("test llm response on head 0:")
    row = df.iloc[0]

    prompt = generate_prompt(
        row["identity_A"],
        row["identity_B"],
        row["scenario_id"]
    )

    print(prompt)

if __name__ == '__main__':
    test_read_parquet()
