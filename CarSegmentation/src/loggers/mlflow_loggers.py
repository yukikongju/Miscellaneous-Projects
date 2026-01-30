"""
This file contains the mlflow logger
"""

import mlflow
import os

from utils.registry import register, LOGGER_REGISTRY
from schemas.loggers.mlflow_loggers import MLFlowLoggersConfig
from mlflow.exceptions import MlflowException


@register(LOGGER_REGISTRY, "mlflow")
class MLFlowLogger:

    def __init__(self, cfg: MLFlowLoggersConfig):
        self.tracking_uri = cfg["tracking_uri"]
        self.experiment_name = cfg["experiment_name"]
        self.num_epochs = cfg["num_epochs"]

        self._init_mlflow()

    def _init_mlflow(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            mlflow.set_experiment(self.experiment_name)
        except Exception as e:
            raise MlflowException(
                f"Couldn't set tracking uri and experiment name properly. Exiting with error {e}"
            )

    def log_metric(self, key: str, value: str, epoch: int = None):
        pass

    def log_config(self):
        pass

    def log_model(self):
        pass
