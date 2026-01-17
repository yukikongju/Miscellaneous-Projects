import torch
from schemas.optimizers.adam_config import AdamOptimizerConfig
from utils.registry import register, OPTIMIZER_REGISTRY


@register(OPTIMIZER_REGISTRY, "adam")
def build_adam(cfg: AdamOptimizerConfig):
    return lambda params: torch.optim.Adam(params, lr=cfg["lr"])
