"""
This file contains utils to load yaml config.
"""

import os
import yaml


def load_config(config_path: str):
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file {config_path} doesn't exist. Please check!")

    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    cfg = {}
    for entry in base["defaults"]:
        for key, value in entry.items():
            cfg[key] = {"name": value}
            for k in base.get(key, {}):
                cfg[key].update({k: base[key][k]})

    return cfg
