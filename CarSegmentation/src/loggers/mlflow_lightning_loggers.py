import os

from lightning.pytorch.loggers import MLFlowLogger
from utils.registry import register, LOGGER_REGISTRY
from schemas.loggers.mlflow_lightning_loggers import MLFlowLightningLoggerConfig


@register(LOGGER_REGISTRY, "mlflow_lightning")
def get_mlflow_logger(cfg: MLFlowLightningLoggerConfig):
    return MLFlowLogger(
        experiment_name=cfg["experiment_name"],
        tracking_uri=cfg["tracking_uri"],
        log_model=cfg["log_model"],
    )
