# base configurations for experiments
from pydantic import BaseModel


class ModelConfig(BaseModel):
    input_channels: int
    num_classes: int


class OptimizerConfig(BaseModel):
    lr: float = 1e-3


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class Config(BaseModel):
    save_dir: str | None = "checkpoints/default"
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
