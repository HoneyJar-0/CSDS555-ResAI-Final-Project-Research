import os
import re
import gc
from datetime import datetime

import psutil
import torch

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from unsloth import FastLanguageModel

from data_loader import PromptDataLoader
from parquet_handler import BatchWriter

from configs import experiment_config

process = psutil.Process(os.getpid())

class Benchmark:
    def __init__(self, start_uuid=0, end_uuid=50):
        self.model_name = self.get_model_name(experiment_config.model_id)
        self.model, self.tokenizer = self.load_model()

        data = PromptDataLoader(start_uuid=start_uuid, end_uuid=end_uuid)
        self.loader = data.load_parquet_to_df(batch_size=experiment_config.batch_size)
        self.writer = BatchWriter()

        if experiment_config.tensorboard_active:
            self.tensorboard = SummaryWriter(log_dir=os.path.join(experiment_config.log_dir, datetime.now().strftime(experiment_config.log_format)))

    def get_model_name(self, model_id: str) -> str:
        model_dict = {
            "llama": "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
            "gemma": "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
            "mistral": "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
        }
        if model_id.strip().lower() not in model_dict.keys():
            raise ValueError(f"Invalid Model Name, expected {list(model_dict.keys())} got {model_id}")

        return model_dict[model_id]

    def load_model(self):
        """Loads model_id"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = experiment_config.max_seq_length,
            dtype = experiment_config.model_dtype,
            load_in_4bit = experiment_config.load_in_4bit,
        )

        tokenizer.padding_side = experiment_config.padding_side
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = experiment_config.padding_side

        return model, tokenizer

    def extract_responses(self, outputs: list) -> list:
        """
        Extract only the text generated in the ### Response: section from each output.
        """
        responses = []

        for out in outputs:
            text = out

            # Remove all special tokens like <|begin_of_text|>, <|finetune_right_pad_id|>
            text = re.sub(r"<\|.*?\|>", "", text)

            text = re.sub(r"</s>|<s>", "", text)

            # Extract everything after the last ### Response:
            if "### Response:" in text:
                # Take text after the last occurrence of ### Response:
                text = text.split("### Response:")[-1]

            # Remove everything after the next ### Input (if any)
            text = text.split("### Input:")[0]

            # Strip leading/trailing whitespace
            text = text.strip()

            responses.append(text)

        return responses

    def run(self):
        """
        Runs the benchmark
        """
        # Load the model for inferencing
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

        for i,batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="Processing Batches"):
            # Create prompts for inputs for batch
            inputs = self.tokenizer(
                batch[1],
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to("cuda")

            # Inference on batch
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=experiment_config.max_new_tokens,
                do_sample=False,
                temperature=experiment_config.temperature,
                use_cache=True
            )

            decoded = self.tokenizer.batch_decode(outputs)

            # Post process batch
            processed_batch = self.extract_responses(decoded)

            # Store aligned with original IDs
            for id, out in zip(batch[0], processed_batch):
                self.writer.add(id, out)

            if experiment_config.tensorboard_active and i%experiment_config.log_interval == 0:
                self.tensorboard.add_scalar("Progress/Percentage", i/len(self.loader)*100, i)
                self.tensorboard.add_scalar("Current UUID", batch[0][0], i)
                self.tensorboard.add_scalar("GPU VRAM Allocated (GB)", torch.cuda.memory_allocated(0) / 1e9, i)
                self.tensorboard.add_scalar("CPU RAM Allocated (GB)", process.memory_info().rss / 1e9, i)

        # Final flush to save remaining data
        out_path = self.writer.flush()

        return out_path

def pipeline(config):
    runner = Benchmark(start_uuid=config["start"], end_uuid=config["end"])
    results = runner.run()
    print(results)

if __name__ == "__main__":
    s_id = 0
    e_id = 10

    runner = Benchmark(start_uuid=s_id, end_uuid=e_id)
    results = runner.run()
