"""
This file contains the Base Model to verify the Torch Trainer with pydantic
"""

from pydantic import BaseModel


class TorchTrainerConfig(BaseModel):
    num_epochs: int
    device: str
    log_every_n_steps: int
