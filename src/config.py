# base configurations for experiments
from collections.abc import Callable

from pydantic import BaseModel


class LoggerConfig(BaseModel):
    enabled: bool = True
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []
    log_every_n_step: int = 1


class ModelConfig(BaseModel):
    input_channels: int
    num_classes: int


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss_name: str


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class DataConfig(BaseModel):
    root_dir: str
    raw_data_dir: str
    train_split: str
    val_split: str
    test_split: str

    batch_size: int = 64
    filename_col: str = "filename"
    num_workers: int = 0
    transform: Callable | None = None

    output_normalization: str = "min_max"  # options: min_max for approach 2, prob for approach 1
    feature_names_list: list[str] = ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "wind_grid"]
    fuel_feats_encoding: str  # ordinal, one_hot
    normalize_fuel_feats_ordinal: bool = True
    valid_mask_threshold: float = 0.00


class Config(BaseModel):
    save_dir: str = "experiments/default"
    base_dir: str = "../yan_bp3"
    modelling_approach: str = "2"
    model: ModelConfig
    optimizer: OptimizerConfig
    training: TrainingConfig
    data: DataConfig
    logger: LoggerConfig
    metrics: list[str] = ["mse", "mae", "spearman", "ssim"]
