from dataset_pipeline import identities as identity_script
import os
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from time import time

from configs import experiment_config

df_id = pd.read_csv(f'{experiment_config.input_dir}/identities.csv')
df_id.sort_values("id").reset_index(drop=True)
identities = df_id.set_index("id")["identity"].to_dict()

df_scen = pd.read_csv(f'{experiment_config.input_dir}/scenarios.csv')
scenarios = df_scen.set_index("id")["scenario"].to_dict()

def create_dataset():
    if not os.path.isfile('./data/input/identities.csv'):
        umbrella, gender, so, ro = identity_script.get_queer_attributes()
        identity_script.save_identities_to_file(identity_script.attribute_pairing(umbrella, gender, so, ro))

    identity_ids = df_id["id"].tolist()

    df_scenario = pd.read_csv(f'{experiment_config.input_dir}/scenarios.csv')
    df_scenario.sort_values("id").reset_index(drop=True)
    scenario_ids = df_scenario["id"].tolist()

    uuid = 0
    time_start = time()

    writer = None
    rows = []

    for sys_identity in identity_ids:
        for scen_identity in identity_ids:
            for scenario in scenario_ids:
                rows.append({
                    "UUID": uuid,
                    "identity_A": sys_identity,
                    "identity_B": scen_identity,
                    "scenario_id": scenario
                })

                if len(rows) >= 1000:
                    df = pd.DataFrame(rows)
                    table = pa.Table.from_pandas(df)

                    if writer is None:
                        writer = pq.ParquetWriter(f"{experiment_config.input_dir}/dataset.parquet", table.schema)
                    writer.write_table(table)

                    rows = []

                if uuid % 1000 == 0:
                    progress = (10 * uuid) // (len(identity_ids)**2 * len(scenario_ids))
                    t = time() - time_start
                    print(f"[{'='*progress}>{' '*(10-progress)}] {progress*10}% {int(t//60)}min {int(t%60)}s", end="\r")

                uuid += 1

    if rows:
        df = pd.DataFrame(rows)
        table = pa.Table.from_pandas(df)
        if writer is None:
            writer = pq.ParquetWriter(f"{experiment_config.input_dir}/dataset.parquet", table.schema)
        writer.write_table(table)

    if writer:
        writer.close()

def split_parquet(in_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    reader = pq.ParquetFile(in_path)
    target_bytes = 90 * 1024 * 1024

    file_idx = 0
    buffered_tables = []
    buffered_bytes = 0

    for rg_idx in range(reader.num_row_groups):
        rg_table = reader.read_row_group(rg_idx)

        rg_size = rg_table.nbytes

        if buffered_bytes + rg_size > target_bytes and buffered_tables:
            out_path = os.path.join(out_dir, f"part_{file_idx}.parquet")
            pq.write_table(pa.concat_tables(buffered_tables), out_path)

            file_idx += 1
            buffered_tables = []
            buffered_bytes = 0

        buffered_tables.append(rg_table)
        buffered_bytes += rg_size

    if buffered_tables:
        out_path = os.path.join(out_dir, f"part_{file_idx}.parquet")
        pq.write_table(pa.concat_tables(buffered_tables), out_path)

def pipeline():
    create_dataset()
    split_parquet(f"{experiment_config.input_dir}/dataset.parquet", f"{experiment_config.input_dir}/dataset")

    # Remove big dataset
<<<<<<< HEAD
    if os.path.exists(f"{experiment_config.input_dir}/dataset.parquet"):
        os.remove(f"{experiment_config.input_dir}/dataset.parquet")
=======
    if os.path.exists("./data/input/dataset.parquet"):
        os.remove("./data/input/dataset.parquet")

if __name__ == "__main__":
    pipeline()
>>>>>>> 4931326 (modified main.py to adhere to new system)
