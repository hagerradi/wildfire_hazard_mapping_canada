# base configurations for experiments
from pydantic import BaseModel


class LoggerConfig(BaseModel):
    enabled: bool = True
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []
    log_every_n_step: int = 1


class ModelConfig(BaseModel):
    num_classes: int = 1
    hidden_features: list[int] = [64, 128, 256, 512]

    # Controls if we use MultiSourceUNet or BaselineUNet
    # Use ["spatial"] for base unet
    # Extra tabular features are detected automatically from the dataset config.
    input_feature_list: list[str] = ["spatial"]

    # encoder/decoder
    use_skip_connections: bool = True
    use_transpose_conv: bool = False
    use_activation_after_upsampling: bool = False

    # specific to tabular model
    tabular_embed_dim: int = 64
    tabular_pooling: str = "max"


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss: str | list[str]
    loss_weights: dict[str, float] = {}


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class EvaluationConfig(BaseModel):
    best_ckpt_metrics: list[str] = ["spearman", "ssim"]  # metric to choose best checkpoint
    best_ckpt_metrics_mode: list[str] = ["max", "max"]  # max, or min
    checkpoint_filename: str = "last.pth"


class GridParams(BaseModel):
    """Specific parameters for the GridSource."""

    feature_names_list: list[str]
    out_norm: str = "min_max"
    fuel_feats_encoding: str = "one_hot"
    normalize_fuel_feats_ordinal: bool = True
    transforms_list: list[str]
    augmentation_prob: float


class TabularParams(BaseModel):
    """Specific parameters for any tabular source (e.g., weather)."""

    csv_name: str = "weather_table.csv"
    feature_names_list: list[str]
    sampling_approach: str = "mode"
    num_samples_per_patch: int = 128
    transforms_list: list[str]
    augmentation_prob: float


class DataSourceConfig(BaseModel):
    name: str
    params: GridParams | TabularParams


class DataConfig(BaseModel):
    root_dir: str
    raw_data_dir: str
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
