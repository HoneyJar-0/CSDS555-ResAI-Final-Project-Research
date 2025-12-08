from configs import experiment_config
import pandas as pd
from llm_pipeline.data_loader import PromptDataLoader
from dataset_pipeline import identities

def verify_uuid_match():
    try:
        directory = f"{experiment_config.eval_dir}/attribute_filter"
        um_df = pd.read_csv(f"{directory}/umbrella_idx.csv")
        so_df = pd.read_csv(f"{directory}/so_idx.csv")
        gen_df = pd.read_csv(f"{directory}/gender_idx.csv")
    except:
       um_df, so_df, gen_df = identities.get_ranges()
    
    id_list = pd.read_csv(f"{experiment_config.input_dir}/indentities.csv")
    scen_list = pd.read_csv(f"{experiment_config.input_dir}/scenarios.csv")
    uuid_per_sys_id = len(id_list)*len(scen_list)
    print(uuid_per_sys_id, 593*4)

if __name__ == '__main__':
    verify_uuid_match()