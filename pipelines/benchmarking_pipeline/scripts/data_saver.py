import os
import time
import pandas as pd

class BatchWriter:
    def __init__(self, model_name, worker_name, output_dir="./data/output", buffer_size=100):
        self.output_dir = os.path.join(output_dir, f"model_{model_name.strip()}", f"user_{worker_name.strip()}")
        
        self.buffer = []
        self.buffer_size = buffer_size
        self.file_counter = 0
        self.model_name = model_name
        
        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, uuid, response):
        self.buffer.append({
            "UUID": uuid, 
            "response": response, 
            "model": self.model_name
        })
        
        if len(self.buffer) % self.buffer_size == 0:
            self._check_size()

    def _check_size(self):
        df_temp = pd.DataFrame(self.buffer)
        if df_temp.memory_usage(deep=True).sum() >= 90 * 1024 * 1024:
            print("Memory Full, Flushing Data")
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        filename = f"part_{self.file_counter:04d}_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
        filepath = os.path.join(self.output_dir, filename)

        try:
            pd.DataFrame(self.buffer).to_parquet(filepath, index=True)
            print(f"Flushed to: {filepath}")
        except Exception as e:
            print(f"Error flushing data to {filepath}: {e}")

        self.buffer = []
        self.file_counter += 1

        return filepath


if __name__ == '__main__':
    writer = BatchWriter(
        output_dir="/workspaces/Project/CSDS555-ResAI-Final-Project-Research/data/output", 
        model_name="gpt-4",
        worker_name="Daniel"
    )

    for i in range(10000):
        writer.add(uuid=i, response=f"Response from GPT-4 for item {i}")

    writer.flush()
