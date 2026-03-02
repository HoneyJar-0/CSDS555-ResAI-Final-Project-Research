import os
import gc
from datetime import datetime

import torch
import psutil
import port_forwarding

from tqdm import tqdm
from vllm import LLM, SamplingParams
from torch.utils.tensorboard import SummaryWriter

from parquet_handler import BatchWriter
from llm_pipeline.data_loader import PromptDataLoader

from configs import experiment_config

process = psutil.Process(os.getpid())

class Benchmark:
    def __init__(self):
        self.model_name = self.get_model_name(experiment_config.model_id)
        self.model = self.load_model()
        self.sampling_params = SamplingParams(
            temperature=experiment_config.temperature,
            max_tokens=experiment_config.max_new_tokens
        )

        data = PromptDataLoader()
        self.loader = data.load_parquet_to_df(batch_size=experiment_config.batch_size)
        self.writer = BatchWriter()

        if experiment_config.tensorboard_active:
            self.tensorboard = SummaryWriter(log_dir=os.path.join(experiment_config.log_dir, datetime.now().strftime(experiment_config.log_format)))
            port_forwarding.launch_tensorboard()


    def get_model_name(self, model_id: str) -> str:
        model_dict = {
            "llama": "meta-llama/Llama-3.1-8B-Instruct",
            "gemma": "google/gemma-3-12b-it",
            "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
            "qwen": "Qwen/Qwen3-8B"
        }
        if model_id.strip().lower() not in model_dict.keys():
            raise ValueError(f"Invalid Model Name, expected {list(model_dict.keys())} got {model_id}")

        return model_dict[model_id]

    def load_model(self) -> LLM:
        """Loads model_id"""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        model = LLM(
            model=self.model_name,
            enforce_eager=True,  # TODO: Disables CUDA graphs which records history of sorts
            dtype="auto",
            # dtype=experiment_config.model_dtype,  # TODO: different formatting than what we have.
            max_model_len=experiment_config.batch_size,
            max_num_seqs=experiment_config.max_seq_length,
            quantization= "bitsandbytes" if experiment_config.load_in_4bit else None,
        )

        return model

    def run(self) -> str:
        """
        Runs the benchmark
        """
        for i, batch in tqdm(enumerate(self.loader), total=len(self.loader), desc="Processing Batches"):
            _, chat_prompts = batch

            # Inference on batch
            outputs = self.model.generate(
                prompts=chat_prompts,
                sampling_params=self.sampling_params,
                use_tqdm=False
            )

            # Store aligned with original IDs
            notice_id = 0
            for id, out in zip(batch[0], outputs):
                self.writer.add(id, out.outputs[0].text.strip())
                notice_id = id

            if experiment_config.tensorboard_active and i%experiment_config.log_interval == 0:
                self.tensorboard.add_scalar("Progress/Percentage", i/len(self.loader)*100, i)
                self.tensorboard.add_scalar("Current UUID", batch[0][0], i)
                self.tensorboard.add_scalar("GPU VRAM Allocated (GB)", torch.cuda.memory_allocated(0) / 1e9, i)
                self.tensorboard.add_scalar("CPU RAM Allocated (GB)", process.memory_info().rss / 1e9, i)
            print(f"Flushed batch {i} to {self.writer.flush()}")
            with open('./notice.txt', 'w')as fp:
                fp.write(str(notice_id))
        # Final flush to save remaining data
        out_path = self.writer.flush()
        return out_path

def pipeline():
    benchmark = Benchmark().run()

if __name__ == "__main__":
    pipeline()
