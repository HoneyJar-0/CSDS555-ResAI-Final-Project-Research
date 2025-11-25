import os
import re
import gc
from datetime import datetime

import torch
import psutil
import pandas as pd

from tqdm import tqdm
from unsloth import FastLanguageModel
from torch.utils.tensorboard import SummaryWriter

from parquet_handler import BatchWriter
from llm_pipeline.data_loader import PromptDataLoader

from configs import experiment_config

process = psutil.Process(os.getpid())

class Benchmark:
    def __init__(self):
        self.model_name = self.get_model_name(experiment_config.model_id)
        self.model, self.tokenizer = self.load_model()

        data = PromptDataLoader()
        self.loader = data.load_parquet_to_df(batch_size=experiment_config.batch_size)
        self.writer = BatchWriter()

        if experiment_config.tensorboard_active:
            self.tensorboard = SummaryWriter(log_dir=os.path.join(experiment_config.log_dir, datetime.now().strftime(experiment_config.log_format)))

    def get_model_name(self, model_id: str) -> str:
        model_dict = {
            "llama": "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
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

        tokenizer.padding_side = "left"
        tokenizer.truncation_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        if "gemma" in self.model_name:
            from unsloth.chat_templates import get_chat_template
            tokenizer = get_chat_template(
                tokenizer,
                chat_template = "gemma-3",
            )

        return model, tokenizer

    def run(self) -> str:
        """
        Runs the benchmark
        """
        # Load the model for inferencing
        FastLanguageModel.for_inference(self.model)  # Enable native 2x faster inference

        for i,batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="Processing Batches"):
            chat_prompts = [
                self.tokenizer.apply_chat_template(
                    conv,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for conv in batch[1]
            ]
            
            # Create prompts for inputs for batch
            inputs = self.tokenizer(
                chat_prompts,
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
            cleaned = []
            if "mistral" in self.model_name.lower():
                cleaned = [r.replace("[INST]", "").replace("<s>", "").replace("</s>", "").strip().split("[/INST]")[-1] for r in decoded]
            elif "llama" in self.model_name.lower():
                for r in decoded:
                    cleaned_str = r.split("<|start_header_id|>assistant<|end_header_id|>")[1]
                    cleaned_str = cleaned_str.split("<|eot_id|>")[1] if cleaned_str.split("<|eot_id|>")[1] else cleaned_str.split("<|eot_id|>")[0]
                    cleaned.append(cleaned_str)
                #cleaned = [r.split("<|start_header_id|>assistant<|end_header_id|>")[1].split("<|eot_id|>")[0].strip() for r in decoded]
            if not cleaned:
                raise ValueError(f"Empty output. Check Decoded:\n{decoded}")

            # Store aligned with original IDs
            for id, out in zip(batch[0], cleaned):
                self.writer.add(id, out.strip())

            if experiment_config.tensorboard_active and i%experiment_config.log_interval == 0:
                self.tensorboard.add_scalar("Progress/Percentage", i/len(self.loader)*100, i)
                self.tensorboard.add_scalar("Current UUID", batch[0][0], i)
                self.tensorboard.add_scalar("GPU VRAM Allocated (GB)", torch.cuda.memory_allocated(0) / 1e9, i)
                self.tensorboard.add_scalar("CPU RAM Allocated (GB)", process.memory_info().rss / 1e9, i)

        # Final flush to save remaining data
        out_path = self.writer.flush()
        return out_path

def pipeline():
    benchmark = Benchmark().run()
    df = pd.read_parquet(benchmark)
    for r in df["response"]:
        print(r)

if __name__ == "__main__":
    pipeline()
