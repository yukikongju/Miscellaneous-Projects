from pydantic import BaseModel


class MLFlowLoggerConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
