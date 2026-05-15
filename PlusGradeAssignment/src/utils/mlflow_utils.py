"""
MLflow logging helpers for model artifacts and feature importances.

Usage:
    from utils.mlflow_utils import log_feature_importances, log_model_artifact
    log_feature_importances(model, feature_names)
    log_model_artifact(model, "model_m1", input_example=X_val[:5])

Caveats:
    - Must be called inside an active mlflow.start_run() context.
    - log_feature_importances silently skips models that expose neither
      feature_importances_ nor coef_.
"""

import mlflow
import mlflow.sklearn


def log_feature_importances(model, feature_names: list[str]) -> None:
    """Log per-feature importances as MLflow metrics."""
    importances = model.feature_importances()
    if importances is None:
        return
    for name, val in zip(feature_names, importances):
        mlflow.log_metric(f"importance_{name}", float(abs(val)))


def log_model_artifact(model, artifact_name: str, input_example=None) -> None:
    """Log the fitted sklearn-compatible estimator as an MLflow artifact."""
    mlflow.sklearn.log_model(model.estimator, artifact_name, input_example=input_example)
