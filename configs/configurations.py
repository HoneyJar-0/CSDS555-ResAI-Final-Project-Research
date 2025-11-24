from dataclasses import dataclass, is_dataclass
import yaml


@dataclass
class Config:
    data_dir: str = "data"
    log_dir: str = "data/logs"
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    log_format: str = "%Y-%m-%d_%H%M%S"
    models_dir: str = "data/models"

@dataclass
class TestConfig(Config):
    # Model IDs
    model_id: str = ""
    # Generate
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_p: float = 0.9
    top_k: int = 50
    tensorboard_active: bool = False
    tensorboard_port: int = 6006
    log_steps: int = 10
    log_limit: int = 3

_converters = {
    #"optimizer_cls": lambda x: getattr(torch.optim, x),
}

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
