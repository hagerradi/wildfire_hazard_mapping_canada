from pathlib import Path

import yaml

from src.config import Config, SpatializedTabularParams
from src.utils import AVAILABLE_METRICS, build_single_loss

SPATIALIZED_CONFIGS = [
    Path("configs/default_v1_full_data_fi_full_config_coordconv_spatialized_weather_fire_size_missing_mask.yaml"),
    Path("configs/default_v1_full_data_ros_full_config_coordconv_spatialized_weather_fire_size_missing_mask.yaml"),
]


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


def test_spatialized_fi_ros_configs_reference_supported_losses_and_metrics():
    for path in SPATIALIZED_CONFIGS:
        config = _load_config(path)
        loss_names = config.optimizer.loss if isinstance(config.optimizer.loss, list) else [config.optimizer.loss]

        for loss_name in loss_names:
            build_single_loss(loss_name, huber_beta=config.optimizer.huber_beta)
        for metric_name in config.metrics:
            assert metric_name in AVAILABLE_METRICS
