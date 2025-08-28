import bentoml
import mlflow
import os
import ssl

from datetime import datetime
from dotenv import load_dotenv
from typing import Dict
from mlflow.models import infer_signature
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import ClassifierMixin

ssl._create_default_https_context = ssl._create_unverified_context


def get_mlflow_experiment(experiment_name: str):
    if not mlflow.get_experiment_by_name(name=experiment_name):
        mlflow.create_experiment(name=experiment_name)
    experiment = mlflow.get_experiment_by_name(experiment_name)
    return experiment


def train_model(params: Dict[str, int | str]) -> KNeighborsClassifier:
    iris = load_iris()
    X_train, Y_train = iris.data[:, :4], iris.target
    model = KNeighborsClassifier(**params)
    model.fit(X_train, Y_train)
    return model


def log_sklearn_model(experiment, model: ClassifierMixin, params: Dict[str, int | str]):
    with mlflow.start_run(
        experiment_id=experiment.experiment_id,
        run_name=f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ):
        for param, val in params.items():
            mlflow.log_param(param, val)

        result = mlflow.sklearn.log_model(sk_model=model, artifact_path=MODEL_NAME)
        return result


def train():
    experiment = get_mlflow_experiment(EXPERIMENT_NAME)

    params = {"n_neighbors": 5, "weights": "distance", "algorithm": "kd_tree"}
    model = train_model(params)
    result = log_sklearn_model(experiment, model, params)
    print(result.model_uri)


def save():
    RUN_ID = "d038e08907df4870a2217d8b3462e29a"
    model_uri = f"runs:/{RUN_ID}/{MODEL_NAME}"
    bentoml.mlflow.import_model("iris", model_uri)


if __name__ == "__main__":
    EXPERIMENT_NAME = "iris"
    MODEL_NAME = "kneighbors_model"

    load_dotenv()
    MLFLOW_URI = os.getenv("MLFLOW_URI")
    if MLFLOW_URI:
        mlflow.set_tracking_uri(MLFLOW_URI)

    #  train()
    save()
