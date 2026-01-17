from pydantic import BaseModel


class AdamOptimizerConfig(BaseModel):
    lr: float
