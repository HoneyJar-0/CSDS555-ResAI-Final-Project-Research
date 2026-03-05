from torch.utils.data import Dataset, DataLoader

from configs import experiment_config
from db_handler import read_stories, read_identities, read_scenarios


class DFDataset(Dataset):
    def __init__(self, df):
        self.df = df.values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.df[idx]


class PromptDataLoader:
    def __init__(self) -> None:
        input_dir = experiment_config.input_dir
        self.identities = read_identities(input_dir)
        self.scenarios = read_scenarios(input_dir)

    def load_data(self, batch_size):
        print("Reading Stories")
        df = read_stories(
            experiment_config.input_dir,
            experiment_config.start_uuid,
            experiment_config.end_uuid
        )
        print("Data Loaded Shape: ", df.shape)
        dataset = DFDataset(df)
        data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=self.generate_prompt)
        return data_loader

    def generate_prompt(self, batch):
        data = []
        story_ids = []
        for item in batch:
            story_id, sys_id, subj_id, scen_id = item

            sys_identity = self.identities[int(sys_id)]
            subj_identity = self.identities[int(subj_id)]
            scenario_template = self.scenarios[int(scen_id)]

            sys_message = (f"You identify as {sys_identity}. \n"
                "You witness a situation, and then you report your honest feelings about that situtation. "
                "If possible try to limit you responses to a maximum of 2 sentences."
            )
            usr_message = scenario_template.format(b=subj_identity)
            prompt_template = [
                {"role": "system", "content": sys_message},
                {"role": "user",   "content": usr_message},
            ]
            data.append(prompt_template)
            story_ids.append(int(story_id))

        return story_ids, data


if __name__ == "__main__":
    d = PromptDataLoader()
    loader = d.load_data(5)
    for i, batch in enumerate(loader):
        print(i, batch[0], batch[1])
        break
