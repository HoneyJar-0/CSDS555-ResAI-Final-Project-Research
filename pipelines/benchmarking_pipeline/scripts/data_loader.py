import os
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class DFDataset(Dataset):
    def __init__(self, df):
        self.df = df.values  # keep values exactly as-is

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]   # returns raw row as numpy array

class PromptDataLoader:
    def __init__(self, start_uuid, end_uuid, path="/workspaces/Project/CSDS555-ResAI-Final-Project-Research/data") -> None:
        """
        Path to data folder
        start_uuid starts from 0
        """
        self.data_path = path
        self.uuid = start_uuid
        self.end_uuid = end_uuid

        df_id = pd.read_csv(f'{path}/input/identities.csv')
        self.identities = df_id.set_index("id")["identity"].to_dict()

        df_scen = pd.read_csv(f'{path}/input/scenarios.csv')
        self.scenarios = df_scen.set_index("id")["scenario"].to_dict()

    def load_parquet_to_df(self, batch_size):
        print("Reading Data")
        parq_par = os.path.join(self.data_path, "./input/dataset/")
        df = pd.read_parquet(
            parq_par,
            filters=[("UUID", ">=", self.uuid), ("UUID", "<", self.end_uuid)]
        )
        print("Data Loaded Shape: ", df.shape)
        dataset = DFDataset(df)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.generate_prompt)
        return data_loader

    def generate_prompt(self, batch):
        alpaca_prompt = (
            "### Instruction: {}\n"
            "### Input: {}\n"
            "### Response:\n {}\n"
        )
        data = []
        uuid_list = []
        # print(f"Shape: {len(batch)}")
        # print(f"Batch: {batch}")
        for item in batch:
            u_id, a_id, b_id, c_id = item

            a = self.identities[a_id]
            b = self.identities[b_id]
            scenario_template = self.scenarios[c_id]

            sys_message = (f"You are a {a}. \n" 
                "Please react to the following scenario paying attention to the role of the individual in the scenario."
                "Use as many adjectives as possible when stating your reactions. Answer in no more than two sentences."
            )
            usr_message = scenario_template.format(b=b)
            msg = alpaca_prompt.format(sys_message, usr_message, "")
            data.append(msg)
            uuid_list.append(int(u_id))

        return uuid_list, data

if __name__ == "__main__":
    d = PromptDataLoader(0, 100)
    loader = d.load_parquet_to_df(5)
    for i, batch in enumerate(loader):
        print(i, batch[0], batch[1])
        break
