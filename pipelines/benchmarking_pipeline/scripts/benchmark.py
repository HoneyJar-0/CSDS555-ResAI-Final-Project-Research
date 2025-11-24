import re
import gc
import torch

from tqdm import tqdm
from unsloth import FastLanguageModel

from data_loader import PromptDataLoader

class Benchmark:
    def __init__(self, model_id, batch_size, start_uuid=0, end_uuid=50):
        self.model_name = self.get_model_name(model_id)
        self.model, self.tokenizer = self.load_model()
        self.batch_size = batch_size
        data = PromptDataLoader(start_uuid=start_uuid, end_uuid=end_uuid)
        self.loader = data.load_parquet_to_df(batch_size=batch_size)

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

        max_seq_length = 2048  # Choose any! We auto support RoPE Scaling internally!
        dtype = None  # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        load_in_4bit = True  # Use 4bit quantization to reduce memory usage. Can be False.

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = self.model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )

        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.truncation_side = "left"

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
        all_outputs = []

        for batch in tqdm(self.loader, desc="Processing Batches"):
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
                max_new_tokens=128,
                do_sample=False,
                temperature=0.0,
                use_cache=True
            )

            decoded = self.tokenizer.batch_decode(outputs)

            # Post process batch
            processed_batch = self.extract_responses(decoded)

            # Store aligned with original IDs
            for i, out in zip(batch[0], processed_batch):
                all_outputs.append((i, out))  # TODO: Instead of list append store to DB or something

        return all_outputs

def pipeline(config):
    runner = Benchmark(model_id=config["model"], 
                       batch_size=config["batch_size"], 
                       start_uuid=config["start"], 
                       end_uuid=config["end"])  # Model downloading takes 10-ish minutes for first time
    results = runner.run()

    for r in results:
        print(r)

if __name__ == "__main__":
    model_name = "mistral"  # Use llama, mistral or gemma
    batch_size = 8
    s_id = 0
    e_id = 10

    runner = Benchmark(model_id=model_name, batch_size=batch_size, start_uuid=s_id, end_uuid=e_id)  # Model downloading takes 10-ish minutes for first time
    results = runner.run()

    for r in results:
        
