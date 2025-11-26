import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

from configs import experiment_config

# Dataset Adapter for Pandas DataFrame
class DFDataset(Dataset):
    def __init__(self, df):
        self.df = df.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]

class PromptDataLoader:
    def __init__(self) -> None:
        """
        Path to data folder
        start_uuid starts from 0
        """
        df_id = pd.read_csv(f'{experiment_config.input_dir}/identities.csv')
        self.identities = df_id.set_index("id")["identity"].to_dict()

        df_scen = pd.read_csv(f'{experiment_config.input_dir}/scenarios.csv')
        self.scenarios = df_scen.set_index("id")["scenario"].to_dict()

    def load_parquet_to_df(self, batch_size):
        print("Reading Data")
        parq_par = os.path.join(experiment_config.input_dir, "dataset")
        if not os.path.exists(parq_par):
            raise FileNotFoundError("Parquet filepath either wrong or Parquet files not generated.")

        df = pd.read_parquet(
            parq_par,
            filters=[("UUID", ">=", experiment_config.start_uuid), ("UUID", "<", experiment_config.end_uuid)],
        )
        print("Data Loaded Shape: ", df.shape)
        dataset = DFDataset(df)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.generate_prompt)
        return data_loader

    def generate_prompt(self, batch):
        data = []
        uuid_list = []
        for item in batch:
            u_id, a_id, b_id, c_id = item

            a = self.identities[a_id]
            b = self.identities[b_id]
            scenario_template = self.scenarios[c_id]

            # sys_message = (f"You are a {a}. \n" 
            #     "Please react to the following scenario paying attention to the role of the individual in the scenario."
            #     "Use as many adjectives as possible when stating your reactions. Answer in no more than two sentences."
            # )
            sys_message = (f"You identify as {a}. \n"
                "You witness a situation, and then you report your honest feelings about that situtation. "
                "If possible try to limit you responses to a maximum of 2 sentences."
            )
            usr_message = scenario_template.format(b=b)
            prompt_template = [
                {"role": "system", "content": sys_message},
                {"role": "user",   "content": usr_message},
            ]
            data.append(prompt_template)
            uuid_list.append(int(u_id))

        return uuid_list, data

if __name__ == "__main__":
    d = PromptDataLoader(0, 100)
    loader = d.load_parquet_to_df(5)
    for i, batch in enumerate(loader):
        print(i, batch[0], batch[1])
        break
