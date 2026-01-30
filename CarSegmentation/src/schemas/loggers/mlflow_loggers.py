from pydantic import BaseModel


class MLFlowLoggersConfig(BaseModel):
    tracking_uri: str
    experiment_name: str
    run_name: str
    num_epochs: str
