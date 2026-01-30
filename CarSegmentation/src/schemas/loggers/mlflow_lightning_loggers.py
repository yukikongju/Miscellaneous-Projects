from pydantic import BaseModel


class MLFlowLightningLoggerConfig(BaseModel):
    experiment_name: str
    tracking_uri: str
    log_model: bool
