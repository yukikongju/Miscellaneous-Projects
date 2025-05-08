import json
import mlflow
import pandas as pd


class MlflowInferenceHandler:
    def __init__(self, model_name: str, tracking_uri: str, stage="latest"):
        self.model_name = model_name
        self.stage = stage
        self.tracking_uri = tracking_uri
        self.model_uri = f"models:/{self.model_name}/{self.stage}"

        self.set_tracking_uri()
        self.load_model()
        self.load_metadata()

    def set_tracking_uri(self) -> None:
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
        except Exception as e:
            raise ValueError(
                f"Couldn't connect to MLFlow server. Please check if tracking uri is valid: {e}"
            )

    def load_model(self) -> None:
        try:
            self.model = mlflow.pyfunc.load_model(self.model_uri)
        except Exception as e:
            raise ValueError(
                f"Couldn't find model {self.model_name} with stage {self.stage}.  Please check if model stage is properly registered."
            )

    def load_metadata(self):
        self.metadata = mlflow.models.get_model_info(self.model_uri)
        try:
            self.signature = self.metadata._signature_dict
            self.X_cols = [item["name"] for item in json.loads(self.signature["inputs"])]
        except Exception as e:
            raise KeyError(
                f"Model signature doesn't exist in metadata. Did you use 'infer_signature()' when logging the model?"
            )

    def predict(self, df: pd.DataFrame):
        missing_columns = set(self.X_cols) - set(df.columns)
        if missing_columns:
            raise ValueError(
                f"Dataframe is missing feature columns: {missing_columns}. Please verify!"
            )

        predictions = self.model.predict(df[self.X_cols])
        return predictions
