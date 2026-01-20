import os

from lightning.pytorch.loggers import MLFlowLogger
from utils.registry import register, LOGGER_REGISTRY
from schemas.loggers.mlflow_loggers import MLFlowLoggerConfig


@register(LOGGER_REGISTRY, "mlflow")
def get_mlflow_logger(cfg: MLFlowLoggerConfig):
    return MLFlowLogger(experiment_name=cfg["experiment_name"], tracking_uri=cfg["tracking_uri"])
