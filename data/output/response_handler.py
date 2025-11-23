import pandas as pd
import os
import time
import json

class BatchWriter:
    def __init__(self, output_dir, model_name, worker_name):
        self.output_dir = os.path.join(output_dir, f"model={model_name}", f"user={worker_name}")
        
        self.buffer = []
        self.file_counter = 0
        self.model_name = model_name
        
        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, uuid, response, **kwargs):
        self.buffer.append({
            "UUID": uuid, 
            "response": response, 
            "model": self.model_name, 
            **kwargs
        })
        
        if len(self.buffer) % 100 == 0: 
            self._check_size()

    def _check_size(self):
        df_temp = pd.DataFrame(self.buffer)
        if df_temp.memory_usage(deep=True).sum() >= 90 * 1024 * 1024:
            self.flush()

    def flush(self):
        if not self.buffer:
            return

        filename = f"part_{self.file_counter:04d}_{time.strftime('%Y%m%d-%H%M%S')}.parquet"
        filepath = os.path.join(self.output_dir, filename)

        try:
            pd.DataFrame(self.buffer).to_parquet(filepath, index=False)
            print(f"Flushed to: {filepath}")
        except Exception as e:
            print(f"Error flushing data to {filepath}: {e}")

        self.buffer = []
        self.file_counter += 1

def output_pipeline(config):
    writer = BatchWriter(
        output_dir="./data/output/dataset", 
        model_name=config["model"],
        worker_name=config["user"]
    )
    for i in range(config["start"],config["end"]):
        writer.add(uuid=i, response=f"Response from {config["model"]} for item {i}")
        writer.flush()
        with open('../notice.txt','w') as fp:
            fp.write(i)

if __name__ == '__main__':
    config = {}
    with open('../config.json','r') as fp:
        config = json.load(fp)
    output_pipeline(config)

