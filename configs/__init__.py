import logging
import os
from dataclasses import fields
from pathlib import Path

from .configurations import ExperimentConfig, Config, load_config

experiment_config = ExperimentConfig()

BASE_CONFIG_PATH = "configs/config.yaml"

def init_dirs(cfg):
    for field in fields(cfg):
        key = field.name
        value = getattr(cfg, key)
        if value:
            p = Path(value)
            if isinstance(value, (str, Path)) and any(sep in str(value) for sep in ("/", "\\")):
                os.makedirs(p, exist_ok=True)

def init_all_configs():
    print("Initializing Configs")
    global_vars = globals()
    config_map = {
        "experiment_config": ExperimentConfig,
    }

    for var_name, cls in config_map.items():
        instance = load_config(BASE_CONFIG_PATH, cls())
        filepath = f"configs/{var_name}.yaml"
        instance = load_config(filepath, instance)
        global_vars[var_name] = instance
    print("Initializing Directories")
    instance = load_config(BASE_CONFIG_PATH, Config())
    init_dirs(instance)

    print("Checking for update via notice.bak")
    if os.path.exists('./notice.bak'):
        with open('./notice.bak','r') as fp:
            updated_uuid = int(fp.readline())
            experiment_config.start_uuid = updated_uuid
            print(f"Found notice.bak, updated start UUID to {updated_uuid}")

    print("All Configs Initialized")

loggers = {
    "main_log": {"level": logging.DEBUG, "file": "main_log.log"}
}

def init_loggers():
    print("Initializing Loggers")
    log_dir = Path(experiment_config.log_dir)
    logging.basicConfig(handlers=[], force=True)

    formatter = logging.Formatter('[%(asctime)s] [%(levelname)-8s] %(name)s: %(message)s')

    for logger_name, config in loggers.items():
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.setLevel(config["level"])
        logger.propagate = False

        log_file = log_dir / config["file"]
        log_file.write_text("")

        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        file_handler.setLevel(config["level"])

        logger.addHandler(file_handler)

    print("All Loggers Initialized")

init_all_configs()
init_loggers()