import torch
from vllm import LLM
from datasets import load_dataset
import time

def get_dataset():
    """
    Loads and returns a Dataset object
    """
    dataset = load_dataset('Qwen/ProcessBench', split='gsm8k')
    dataset= dataset.select_columns(["problem"])
    dataset = dataset.map(lambda x: {"role": "user", "content": x}) # format prompts for the tokenizer
    
    return dataset["problem"]

def generate_inference(prompts, tokenizer='auto', batch_size=256,quantization='bitsandbytes'):
    """
    Runs a model on given input.
    
    prompts: list of dicts containing prompts
    tokenizer: 'auto'|'slow', auto will try to use a fast tokenizer if one is available for the model, slow otherwise
    batch_size: number of sequences/prompts to pass to the model at a time
    quantization: None|str, method of quantization. Varies from model to model; check its documentation.
    
    Returns: list of outputs from the model.
    """
    #load model
    model = "Qwen/Qwen2.5-1.5B"
    
    llm = LLM(
        model=model,                            # model taken from HuggingFace
        tokenizer=model,                        # what tokenizer to use
        tokenizer_mode=tokenizer,               # default is auto; slow is here only for comparison
        tensor_parallel_size=1,                 # Splits the model over n GPUs to fit larger models. WARNING!!! Greatly increases latency in communication
        quantization=quantization,              # Specify type of quantization method wanted. See vLLM docs for specifics
        gpu_memory_utilization=0.9,             # Default value is 90%; mentioned here in case of OOM errors. Specifies how much VRAM to reserve for the model
        max_num_seqs=batch_size                 # Effectively, the batch size. 256 is the default. Decrease to avoid OOM errors if necessary
    )

    outputs = llm.generate(prompts)
    results = [output.outputs[0].text for output in outputs]
    return results

if __name__ == '__main__':
    prompts = get_dataset()
    print("================\nRunning Optimized Inference")
    sped_up_start = time.time()
    print(f"First Ten Outputs: {generate_inference(prompts=prompts)[0:10]}")
    sped_up_end = time.time()
    
    print("================\nRunning Basic Inference")
    start = time.time()
    print(f"First Ten Outputs: {generate_inference(prompts=prompts,tokenizer='slow',batch_size=1,quantization=None)[0:10]}")
    end = time.time()
    
    print("\n\n========================")
    print(f"Metrics -\nOptimized inference runtime: {(sped_up_end - sped_up_start):.2f}s")
    print(f"Basic inference runtime: {(end - start):.2f}s")
