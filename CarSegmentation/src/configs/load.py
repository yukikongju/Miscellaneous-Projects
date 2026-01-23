"""
This file contains utils to load yaml config.
"""

import os
import re
import yaml


def _resolve_env(value: str):
    """
    Read the environment variable defined in '${env:ENV_VARIABLE}'
    """
    ENV_PATTERN = re.compile(r"\$\{env:([^}]+)\}")
    if isinstance(value, str):
        matching = ENV_PATTERN.fullmatch(value)
        if matching:
            return os.environ[matching.group(1)]
    return value


def load_config(config_path: str):
    """
    Load config from yaml file and resolve to read for environment
    variables if value is defined as '${env:ENV_VARIABLE}'
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"The config file {config_path} doesn't exist. Please check!")

    with open(config_path, "r", encoding="utf-8") as f:
        base = yaml.safe_load(f)

    cfg = {}
    for entry in base["defaults"]:
        for key, value in entry.items():
            cfg[key] = {"name": value}
            for k in base.get(key, {}):
                v = _resolve_env(base[key][k])
                cfg[key].update({k: v})

    return cfg
