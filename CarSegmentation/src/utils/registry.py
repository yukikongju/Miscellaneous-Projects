from typing import Dict, Callable

MODEL_REGISTRY: Dict[str, Callable] = {}
DATALOADER_REGISTRY: Dict[str, Callable] = {}
OPTIMIZER_REGISTRY: Dict[str, Callable] = {}
LOGGER_REGISTRY: Dict[str, Callable] = {}
LOSSES_REGISTRY: Dict[str, Callable] = {}


def register(registry, name):
    def decorator(cls_or_fn):
        registry[name] = cls_or_fn
        return cls_or_fn

    return decorator
