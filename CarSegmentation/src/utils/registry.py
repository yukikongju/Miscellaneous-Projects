MODEL_REGISTRY = {}
DATASET_REGISTRY = {}
OPTIMIZER_REGISTRY = {}
TRAINER_REGISTRY = {}


def register(registry, name):
    def decorator(cls_or_fn):
        registry[name] = cls_or_fn
        return cls_or_fn

    return decorator
