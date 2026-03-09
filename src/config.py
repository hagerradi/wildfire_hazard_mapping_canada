# base configurations for experiments
from pydantic import BaseModel, Field


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

    # specific to auxilary model
    auxilary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] = {"weather": [32, 64]}
    auxilary_embed_dims: dict[str, int] = {"weather": 128}
    auxilary_poolings: dict[str, str] = {"weather": "max"}


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss: str | list[str]
    loss_weights: dict[str, float] = {}


class SchedulerConfig(BaseModel):
    name: str | None = None  # "cosine_warmup", "plateau", "onecycle", "multistep" (or null)

    # params specific to each scheduler
    warmup_epochs: int = 5  # for cosine_warmup
    max_lr: float = 1e-3  # for onecycle
    patience: int = 10  # for plateau
    factor: float = 0.1  # for plateau and multistep
    milestones: list[int] = [30, 40]  # for multistep


class TrainingConfig(BaseModel):
    max_epochs: int = 50
    log_every_n_epoch: int = 1


class EvaluationConfig(BaseModel):
    best_ckpt_metrics: list[str] = ["spearman"]  # metric to choose best checkpoint
    best_ckpt_metrics_mode: list[str] = ["max"]  # max, or min
    checkpoint_filename: str = "best.pth"


class GridParams(BaseModel):
    """Specific parameters for the GridSource."""

    feature_names_list: list[str]
    # TODO: move out_norm outside of grid source config since it's for GT
    out_norm: str = "min_max"
    fuel_feats_encoding: str = "one_hot"
    normalize_fuel_feats_ordinal: bool = True
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0


class TabularParams(BaseModel):
    """Specific parameters for any tabular source relying on mapped weather zone id"""

    csv_name: str = "weather_table.csv"
    feature_names_list: list[str]
    fire_weather_zone_id_col: str = "wx_zone"
    sampling_approach: str = "mode"  # mode or weighted
    num_samples_per_patch: int = 256
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0


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

    input_sources: list[DataSourceConfig]


class Config(BaseModel):
    save_dir: str = "experiments/default"
    base_dir: str = "../yan_bp3"
    seed: int = 42
    deterministic: bool = True
    modelling_approach: str = "2"
    model: ModelConfig
    optimizer: OptimizerConfig
    lr_scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)
    training: TrainingConfig
    evaluation: EvaluationConfig
    data: DataConfig
    logger: LoggerConfig
    metrics: list[str] = ["mse", "mae", "spearman", "ssim"]
