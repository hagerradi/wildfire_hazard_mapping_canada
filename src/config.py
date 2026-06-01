# base configurations for experiments
from typing import Any

from pydantic import BaseModel, Field, model_validator


class LoggerConfig(BaseModel):
    enabled: bool = True
    project_name: str
    workspace: str
    experiment_name: str
    tags: list[str] = []
    log_every_n_step: int = 1


class ModelConfig(BaseModel):
    architecture: str = "auto"
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
    use_coordconv: bool = False
    use_multiscale_global_context: bool = False

    # segmentation_models_pytorch models. Used when architecture is "smp".
    smp_architecture: str = "Unet"
    smp_encoder_name: str = "resnet34"
    smp_encoder_weights: str | None = None
    smp_kwargs: dict[str, Any] = Field(default_factory=dict)

    # specific to auxiliary model
    auxiliary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] = {"weather": [32, 64]}
    auxiliary_embed_dims: dict[str, int] = {"weather": 128}
    auxiliary_feature_encoder_poolings: dict[str, str] = {"weather": "max"}


class OptimizerConfig(BaseModel):
    name: str = "AdamW"
    lr: float = 1e-3
    loss: str | list[str]
    loss_weights: dict[str, float] = {}
    huber_beta: float = Field(default=1.0, gt=0.0)
    tail_loss_percentile: float = Field(default=0.75, gt=0.0, lt=1.0)
    tail_loss_weight: float = Field(default=5.0, ge=1.0)
    quantile: float = Field(default=0.85, gt=0.0, lt=1.0)
    target_losses: dict[str, str] = {}
    target_loss_weights: dict[str, float] = {}


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
    bp_nodata_as_zero: bool = True
    prediction_support_policy: str = "input"


class GridParams(BaseModel):
    """Specific parameters for the GridSource."""

    feature_names_list: list[str]
    target_name: str | list[str] = "bp"
    # TODO: move out_norm outside of grid source config since it's for GT
    out_norm: str = "min_max"
    target_out_norms: dict[str, str] = {}
    target_log_mean: float | None = None
    target_log_std: float | None = None
    target_log_means: dict[str, float] = {}
    target_log_stds: dict[str, float] = {}
    fuel_feats_encoding: str = "one_hot"
    normalize_fuel_feats_ordinal: bool = True
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0
    include_hex_coords: bool = False
    terrain_derivatives: list[str] = Field(default_factory=list)
    terrain_cell_size_m: float = Field(default=100.0, gt=0.0)
    bp_nodata_as_zero: bool = False
    input_mask_policy: str = "input_only"


class TabularParams(BaseModel):
    """Specific parameters for any tabular source relying on mapped weather zone id"""

    csv_name: str = "weather_table_processed.csv"
    feature_names_list: list[str]
    fire_weather_zone_id_col: str = "WeatherZone"
    fire_weather_zone_selection_approach: str = "mode"  # Selection method to determine weather zone to be used for the patch: "mode" (for most common zone) or "weighted" (for frequency-weighted sampling)
    num_samples_per_patch: int = 256
    transforms_list: list[str] = Field(default_factory=list)
    augmentation_prob: float = 0.0
    sampling_bias: str | None = (
        None  # Whether to bias sampling towards high or low values of the feature of interest, or no bias (None, "high_values")
    )
    feature_to_bias: str | None = None  # The feature to bias sampling towards if sampling_bias is not None


class SpatializedTabularParams(TabularParams):
    """Parameters for rasterizing zone-level tabular covariates onto patch pixels."""

    zone_channel_key: str = "firezones_grid"
    aggregation: str = "mean"
    shuffle_lut: bool = False
    shuffle_seed: int = 42
    include_missing_mask: bool = False
    missing_value_strategy: str = "global_mean"


class GlobalContextParams(BaseModel):
    """Parameters for downsampled full-hex context grids."""

    context_dir: str
    num_context_channels: int
    context_filename_template: str = "hex_{hex_id_padded}_global_context.npz"
    metadata_key_col: str = "hex_id"
    include_patch_footprint: bool = True
    patch_height: int = 256
    patch_width: int = 256


class BPPredictionParams(BaseModel):
    """Parameters for appending precomputed BP predictions as a scalar raster channel."""

    prediction_dir: str
    filename_template: str = "hexel_{hex_id_padded}_predicted.tif"
    metadata_key_col: str = "hex_id"
    fill_value: float = 0.0
    clip_min: float | None = 0.0
    clip_max: float | None = 1.0
    validate_alignment: bool = True


class DataSourceConfig(BaseModel):
    name: str
    params: GridParams | TabularParams | SpatializedTabularParams | GlobalContextParams | BPPredictionParams

    @model_validator(mode="before")
    @classmethod
    def parse_params_for_source(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        name = data.get("name")
        params = data.get("params")
        if not isinstance(name, str):
            return data
        if not isinstance(params, dict):
            return data

        param_classes = {
            "grid": GridParams,
            "weather": TabularParams,
            "fire_size": TabularParams,
            "spatialized_weather": SpatializedTabularParams,
            "spatialized_fire_size": SpatializedTabularParams,
            "global_context_grid": GlobalContextParams,
            "bp_prediction": BPPredictionParams,
        }
        param_class = param_classes.get(name)
        if param_class is None:
            return data

        parsed = dict(data)
        parsed["params"] = param_class(**params)
        return parsed


class DataConfig(BaseModel):
    root_dir: str
    raw_data_dir: str
    batch_size: int = 64
    num_workers: int = 0

    train_split: str
    val_split: str
    test_split: str
    filename_col: str = "filename"
    valid_mask_threshold: float = 0.01
    include_patch_metadata: bool = False

    input_sources: list[DataSourceConfig]


class DataPrepConfig(BaseModel):
    modelling_approach: int = 1
    win_h: int = 256
    win_w: int = 256
    overlap_ratio: float = 0.2


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
    data_prep: DataPrepConfig = Field(default_factory=DataPrepConfig)
