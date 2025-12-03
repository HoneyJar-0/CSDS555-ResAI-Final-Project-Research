import pandas as pd
import os
import time
import pyarrow.dataset as ds

from configs import experiment_config


class BatchWriter:
    def __init__(self):
        self.output_dir = os.path.join(experiment_config.output_dir, f"model_{experiment_config.model_id.strip()}", f"user_{experiment_config.worker_name.strip()}")

        self.buffer = []
        self.file_counter = 0

        os.makedirs(self.output_dir, exist_ok=True)

    def add(self, uuid, response):
        self.buffer.append({
            "UUID": uuid,
            "response": response,
            "model": experiment_config.model_id
        })

        if len(self.buffer) % experiment_config.buffer_check_count == 0:
            self._check_size()

    def _check_size(self):
        df_temp = pd.DataFrame(self.buffer)
        if df_temp.memory_usage(deep=True).sum() >= experiment_config.max_parquet_size_mib * 1024 * 1024:
            print("Memory Full, Flushing Data")
            self.flush()

    def flush(self) -> str:
        if not self.buffer:
            return ""

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

class BatchReader:
    def __init__(self, root_dir, filters=None, batch_size=10000):
        self.dataset = ds.dataset(root_dir, format="parquet")
        self.scanner = self.dataset.scanner(filter=filters, use_threads=True).to_batches()
        self.batch_size = batch_size

        self.generator = self._batch_generator()

    def _batch_generator(self):
        buffer = []
        current_rows = 0
        for batch in self.scanner:
            df = batch.to_pandas()
            buffer.append(df)
            current_rows += len(df)

            # Yield when batch size limit is hit
            if current_rows >= self.batch_size:
                yield pd.concat(buffer, ignore_index=True)
                buffer = []
                current_rows = 0

        # Yield remaining data
        if buffer:
            yield pd.concat(buffer, ignore_index=True)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.generator)

    # Iteratively maps over all files without simultaneous loading
    def map(self, function):
        for batch in self.generator:
            yield function(batch)
