from dataclasses import dataclass, is_dataclass

import torch
import yaml


@dataclass
class Config:
    data_dir: str = "data"
    log_dir: str = "data/logs"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    log_format: str = "%Y-%m-%d_%H%M%S"
    models_dir: str = "data/models"
    eval_dir: str = "data/evaluations"
    attrib_dir: str = "data/attributes"

@dataclass
class ExperimentConfig(Config):
    # Model IDs
    worker_name: str = ""
    model_id: str = ""
    start_uuid: int = 0
    end_uuid: int = 0
    # Tokenizer
    max_seq_length: int = 0
    padding_side: str = "left"
    # Generate
    max_new_tokens: int = 0
    batch_size: int = 0
    load_in_4bit: bool = True
    model_dtype: torch.dtype = None
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    # Batching
    buffer_check_count: int = 100
    max_parquet_size_mib: float = 90.0
    # Logging
    tensorboard_active: bool = False
    tensorboard_port: int = 6543
    log_interval: int = 10
    log_limit: int = 3

@dataclass
class EvaluationConfig(Config):
    eval_batch_size: int = 5

_converters = {
    "model_dtype": lambda x: _safe_torch_getattr(x)
}

def _safe_torch_getattr(x):
    try:
        return getattr(torch, x)
    except Exception:
        return None

def _recursive_load(cfg_obj, yaml_dict):
    for key, value in yaml_dict.items():
        if hasattr(cfg_obj, key):
            attr = getattr(cfg_obj, key)
            if key in _converters:
                value = _converters[key](value)
            if is_dataclass(attr) and isinstance(value, dict):
                _recursive_load(attr, value)
            else:
                setattr(cfg_obj, key, value)
    return cfg_obj


def load_config(path: str, cfg: Config) -> Config:
    with open(path, "r") as f:
        yaml_cfg = yaml.safe_load(f)

    cfg = _recursive_load(cfg, yaml_cfg)
    return cfg
