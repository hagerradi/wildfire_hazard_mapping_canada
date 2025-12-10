# base configurations for experiments
from pydantic import BaseModel


class LoggerConfig(BaseModel):
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []


class ModelConfig(BaseModel):
    input_channels: int
    num_classes: int


class OptimizerConfig(BaseModel):
    name: str | None = "AdamW"
    lr: float = 1e-3
    loss_name: str


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class Config(BaseModel):
    save_dir: str = "experiments/default"
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    logger: LoggerConfig
    metrics: list[str] = ["mse", "mae", "spearman", "ssim"]
