# base configurations for experiments
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel


class LoggerConfig(BaseModel):
    enabled: bool = True
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []
    log_every_n_step: int = 1


class ModelConfig(BaseModel):
    num_classes: int


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss: str | list[str]
    loss_weights: dict[str, float] = {}


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class EvaluationConfig(BaseModel):
    best_ckpt_metric: str = "spearman"  # metric to choose best checkpoint
    best_ckpt_metric_mode: str = "max"  # max, or min
    checkpoint_filename: str = "last.pth"


class DataSourceConfig(BaseModel):
    name: str
    params: dict[str, Any] = {}


class DataConfig(BaseModel):
    root_dir: str
    batch_size: int = 64
    num_workers: int = 0

    train_split: str
    val_split: str
    test_split: str
    filename_col: str = "filename"
    valid_mask_threshold: float = 0.0

    sources: list[DataSourceConfig]


class Config(BaseModel):
    save_dir: str = "experiments/default"
    base_dir: str = "../yan_bp3"
    seed: int = 42
    deterministic: bool = True
    modelling_approach: str = "2"
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    data: DataConfig
    logger: LoggerConfig
    metrics: list[str] = ["mse", "mae", "spearman", "ssim"]
