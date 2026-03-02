import time
import gc
import torch
from vllm import LLM, SamplingParams

def main():
    prompts = [
        # "Hello, my name is",
        # "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256)
    print(f"Sampling Params: {sampling_params}")

    model_dict = {
        "llama": "meta-llama/Llama-3.1-8B-Instruct",
        "gemma": "google/gemma-3-12b-it",
        "mixtral": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "qwen": "Qwen/Qwen3-8B"
    }
    quantization = "bitsandbytes"  # None, bitsandbytes
    for model in model_dict.values():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

        print("\n\n", "*"*30, f" {model} ", "*"*30)
        m_time = time.time()
        llm = LLM(
            model=model,                                # model taken from HuggingFace
            enforce_eager=True,                         # Disables CUDA graphs which records history of sorts
            max_model_len=256,
            # tokenizer=model,                          # what tokenizer to use
            # dtype="int4",
            # tokenizer_mode=tokenizer,                 # default is auto; slow is here only for comparison
            # tensor_parallel_size=1,                   # Splits the model over n GPUs to fit larger models. WARNING!!! Greatly increases latency in communication
            quantization=quantization,                  # Specify type of quantization method wanted. See vLLM docs for specifics
            # gpu_memory_utilization=0.9,               # Default value is 90%; mentioned here in case of OOM errors. Specifies how much VRAM to reserve for the model
            max_num_seqs=5                   # Effectively, the batch size. 256 is the default. Decrease to avoid OOM errors if necessary
        )
        load_time = time.time() - m_time

        gen_time = []
        print("\n\n", "="*30, f" {model} ", "="*30)
        for _ in range(3):
            s_time = time.time()
            outputs = llm.generate(prompts, sampling_params)
            gen_time.append(time.time() - s_time)
            
            for output in outputs:
                prompt = output.prompt
                generated_text = output.outputs[0].text
                print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

        print(f"{model} load time: {load_time}")
        print(f"Average Inference time for quantization: {quantization} over 3 runs: {sum(gen_time)/len(gen_time)}")
    print("over")

if __name__ == "__main__":
    main()
