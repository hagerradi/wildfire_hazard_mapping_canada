from pathlib import Path

import yaml

from src.config import Config, GridParams, SpatializedTabularParams
from src.utils import AVAILABLE_METRICS, build_single_loss

SPATIALIZED_CONFIGS = [
    Path("configs/default_v1_full_data_fi_full_config_coordconv_spatialized_weather_fire_size_missing_mask.yaml"),
    Path("configs/default_v1_full_data_ros_full_config_coordconv_spatialized_weather_fire_size_missing_mask.yaml"),
    Path("configs/default_v1_full_data_fi_full_config_coordconv_spatialized_weather_fire_size_missing_mask_terrain.yaml"),
    Path("configs/default_v1_full_data_ros_full_config_coordconv_spatialized_weather_fire_size_missing_mask_terrain.yaml"),
]
BP_CONFIGS = [
    Path("configs/default_v1_full_data_bp_full_config_kl_ccc_hexpairrank.yaml"),
]
COLLEAGUE_READY_CONFIGS = SPATIALIZED_CONFIGS + BP_CONFIGS


def _load_config(path: Path) -> Config:
    with path.open() as f:
        return Config(**yaml.safe_load(f))


def test_spatialized_fi_ros_configs_parse_to_expected_sources():
    for path in SPATIALIZED_CONFIGS:
        config = _load_config(path)
        sources = {source.name: source.params for source in config.data.input_sources}

        assert config.model.input_feature_list == ["spatial"]
        assert config.model.use_coordconv is True
        assert config.evaluation.robust_plot_percentile == 99.0
        assert isinstance(sources["spatialized_weather"], SpatializedTabularParams)
        assert isinstance(sources["spatialized_fire_size"], SpatializedTabularParams)
        assert sources["spatialized_weather"].include_missing_mask is True
        assert sources["spatialized_fire_size"].include_missing_mask is True
        if "terrain" in path.stem:
            assert isinstance(sources["grid"], GridParams)
            assert sources["grid"].terrain_derivatives == ["slope", "aspect_sin", "aspect_cos"]
            assert "terrain_derivatives=slope_aspect" in config.logger.tags


def test_spatialized_fi_ros_configs_reference_supported_losses_and_metrics():
    for path in SPATIALIZED_CONFIGS:
        config = _load_config(path)
        loss_names = config.optimizer.loss if isinstance(config.optimizer.loss, list) else [config.optimizer.loss]

        assert config.data.input_sources[0].params.target_log_mean is None
        assert config.data.input_sources[0].params.target_log_std is None
        assert config.evaluation.best_ckpt_metrics == ["ccc"]
        assert config.evaluation.best_ckpt_metrics_mode == ["max"]
        assert "best_ckpt=ccc" in config.logger.tags
        for loss_name in loss_names:
            build_single_loss(loss_name, huber_beta=config.optimizer.huber_beta)
        for metric_name in config.metrics:
            assert metric_name in AVAILABLE_METRICS


def test_bp_hexpairrank_config_parse_to_expected_sources_and_losses():
    config = _load_config(BP_CONFIGS[0])
    sources = {source.name: source.params for source in config.data.input_sources}

    assert config.model.input_feature_list == ["spatial", "auxiliary"]
    assert config.data.include_patch_metadata is True
    assert set(sources) == {"grid", "tabular_weather", "tabular_fire_size"}
    assert config.optimizer.loss == ["kl", "ccc", "hex_mean_pairwise_rank", "hex_top10_pairwise_rank"]
    assert config.optimizer.loss_weights == {
        "kl": 0.45,
        "ccc": 0.45,
        "hex_mean_pairwise_rank": 0.05,
        "hex_top10_pairwise_rank": 0.05,
    }
    assert "include_patch_metadata=true" in config.logger.tags


def test_colleague_ready_configs_reference_supported_losses_and_metrics():
    for path in COLLEAGUE_READY_CONFIGS:
        config = _load_config(path)
        loss_names = config.optimizer.loss if isinstance(config.optimizer.loss, list) else [config.optimizer.loss]

        assert config.logger.enabled is True
        assert config.logger.log_every_n_step == 10
        assert config.evaluation.robust_plot_percentile == 99.0
        for loss_name in loss_names:
            build_single_loss(loss_name, huber_beta=config.optimizer.huber_beta)
        for metric_name in config.metrics:
            assert metric_name in AVAILABLE_METRICS
