import os
import pandas as pd
from llm_pipeline.data_loader import PromptDataLoader

from configs import experiment_config, evaluation_config

class EvalLoader(PromptDataLoader):
    def __init__(self) -> None:
        super().__init__()

    def load_parquet_to_identity_df(self):
        print("Reading Data")
        parq_par = os.path.join(experiment_config.input_dir, "dataset")
        if not os.path.exists(parq_par):
            raise FileNotFoundError("Parquet filepath either wrong or Parquet files not generated.")

        df = pd.read_parquet(
            parq_par,
            columns=["UUID", "identity_A", "identity_B"]
        )
        print("Data Loaded Shape: ", df.shape)
        print("Data Loaded Shape: ", df.columns)

        return df

    def generate_identity_df(self):
        df = self.load_parquet_to_identity_df()
        df["A"] = df["identity_A"].apply(lambda a_id: self.identities[a_id])
        df["B"] = df["identity_B"].apply(lambda b_id: self.identities[b_id])
        df.drop(labels=["identity_A", "identity_B"], axis=1, inplace=True)
        return df
    
    def load_eval_df(self, model_name:str, text=False):
        if text:
            input_df = self.generate_identity_df()
        else:
            input_df = self.load_parquet_to_identity_df()

        eval_df = pd.read_parquet(
            evaluation_config.eval_dir,
            filters=[("model", "==", model_name.strip().lower())],
            )
        eval_df.drop(columns=["response"], axis=1, inplace=True)
        merged_df = pd.merge(input_df, eval_df, how='inner', on='UUID')

        s_uid, e_uid = merged_df["UUID"].min(), merged_df["UUID"].max()
        save_path = f"{experiment_config.data_dir}/condensed/{model_name}_{s_uid}_{e_uid}.parquet"
        merged_df.to_parquet(save_path, index=True)

        print(f"File saved to: {save_path}")

        return merged_df

if __name__ == "__main__":
    loader = EvalLoader()
    df = loader.load_eval_df(model_name="llama")
    print("Processedd Loaded Shape: ", df.shape)
    print("Processed Loaded Columns: ", df.columns)
    print("Processed Loaded Head: \n", df.head())
    print("Processed Loaded Tail: \n", df.tail())
    print("Total memory (MB):", df.memory_usage(deep=True).sum() / 1024**2)
