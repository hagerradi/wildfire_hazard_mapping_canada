from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.config import (
    Config,
    DataConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.trainer import Trainer

SPATIAL_CHANNELS = 1
WEATHER_FEATS = 5
FIRE_SIZE_FEATS = 3
AUX_SAMPLES = 16  # tabular rows sampled per patch
WIND_CHANNELS = 4
WIND_HEIGHT = 128
WIND_WIDTH = 128


class DummyLoss(torch.nn.Module):
    def forward(self, predictions, targets, masks):
        return torch.tensor(1.0, requires_grad=True)


def dummy_metric(predictions, targets, masks):
    return torch.tensor(0.5)


class GridDataset(torch.utils.data.Dataset):
    """Spatial-only dataset —> yields {'grid': (inputs, targets, masks)}."""

    def __init__(self, size: int = 4, channels: int = 1, height: int = 32, width: int = 32):
        self.size = size
        self.channels = channels
        self.height = height
        self.width = width

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict:
        torch.manual_seed(idx)
        inputs = torch.rand(self.channels, self.height, self.width)
        targets = inputs * 0.9
        masks = (torch.rand(1, self.height, self.width) > 0.5).float()
        return {"grid": (inputs, targets, masks)}


class PatchMetadataDataset(GridDataset):
    """Spatial dataset with the hex IDs needed by hex-summary losses."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        item["patch_metadata"] = {"hex_id": torch.tensor(idx + 1, dtype=torch.long)}
        return item


class WeatherDataset(GridDataset):
    """Spatial + weather tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 1000)
        item["weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        return item


class MultiAuxDataset(GridDataset):
    """Spatial + weather + fire_size tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 1000)
        item["weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        item["fire_size"] = torch.rand(AUX_SAMPLES, FIRE_SIZE_FEATS)
        return item


class WindGridDataset(GridDataset):
    """Spatial + spatial wind grid dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 2000)
        item["wind_grid_mixer"] = torch.rand(WIND_CHANNELS, WIND_HEIGHT, WIND_WIDTH)
        return item


class WindAndWeatherDataset(GridDataset):
    """Spatial + spatial wind grid + weather tabular dataset."""

    def __getitem__(self, idx: int) -> dict:
        item = super().__getitem__(idx)
        torch.manual_seed(idx + 2000)
        item["wind_grid_mixer"] = torch.rand(WIND_CHANNELS, WIND_HEIGHT, WIND_WIDTH)
        torch.manual_seed(idx + 3000)
        item["weather"] = torch.rand(AUX_SAMPLES, WEATHER_FEATS)
        return item


def _make_config(
    tmp_path,
    *,
    logger_enabled: bool = False,
    input_feature_list: list[str] | None = None,
    grid_params: GridParams | None = None,
    auxiliary_hidden_dims: dict | None = None,
    auxiliary_embed_dims: dict | None = None,
    auxiliary_feature_encoder_poolings: dict | None = None,
) -> Config:
    """Config. factory"""
    if input_feature_list is None:
        input_feature_list = ["spatial"]

    return Config(
        save_dir=str(tmp_path),
        model=ModelConfig(
            num_classes=1,
            hidden_features=[8, 16],
            input_feature_list=input_feature_list,
            auxiliary_hidden_dims=auxiliary_hidden_dims or {"weather": [16, 32]},
            auxiliary_embed_dims=auxiliary_embed_dims or {"weather": 16},
            auxiliary_feature_encoder_poolings=auxiliary_feature_encoder_poolings or {"weather": "max"},
        ),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(
            best_ckpt_metrics=["spearman"],
            best_ckpt_metrics_mode=["max"],
            checkpoint_filename="best.pth",
        ),
        data=DataConfig(
            root_dir="",
            raw_data_dir="",
            train_split="",
            val_split="",
            test_split="",
            input_sources=[
                DataSourceConfig(
                    name="grid",
                    params=grid_params or GridParams(feature_names_list=["dummy_feat"]),
                )
            ],
        ),
        logger=LoggerConfig(
            enabled=logger_enabled,
            project_name="test",
            workspace="test",
            experiment_name="test",
        ),
        metrics=["mse", "spearman"],
    )


@pytest.fixture
def dummy_config(tmp_path):
    return _make_config(tmp_path)


@pytest.fixture
def dummy_config_with_logger(tmp_path):
    return _make_config(tmp_path, logger_enabled=True)


@pytest.fixture
def auxiliary_config(tmp_path):
    return _make_config(
        tmp_path,
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"weather": [16, 32]},
        auxiliary_embed_dims={"weather": 16},
        auxiliary_feature_encoder_poolings={"weather": "max"},
    )


@pytest.fixture
def multi_aux_config(tmp_path):
    return _make_config(
        tmp_path,
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"weather": [16, 32], "fire_size": [16, 32]},
        auxiliary_embed_dims={"weather": 16, "fire_size": 16},
        auxiliary_feature_encoder_poolings={"weather": "max", "fire_size": "max"},
    )


@pytest.fixture
def wind_grid_config(tmp_path):
    return _make_config(
        tmp_path,
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"wind_grid_mixer": {"mixer": [16], "local": [32, 64, 16], "global": [16]}},
        auxiliary_embed_dims={"wind_grid_mixer": 16},
        auxiliary_feature_encoder_poolings={"wind_grid_mixer": "max"},
    )


@pytest.fixture
def wind_and_weather_config(tmp_path):
    return _make_config(
        tmp_path,
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_hidden_dims={
            "wind_grid_mixer": {"mixer": [16], "local": [32, 64, 16], "global": [16]},
            "weather": [16, 32],
        },
        auxiliary_embed_dims={"wind_grid_mixer": 16, "weather": 16},
        auxiliary_feature_encoder_poolings={"wind_grid_mixer": "max", "weather": "max"},
    )


@pytest.fixture
def mock_comet_logger(monkeypatch):
    """
    Mock the CometLogger inside src.trainer.
    Apply this fixture explicitly to tests that enable the logger.
    """
    import src.trainer as trainer_module

    mock_logger_cls = MagicMock(name="CometLogger")
    mock_logger_instance = MagicMock(name="logger_instance")
    mock_logger_cls.return_value = mock_logger_instance
    monkeypatch.setattr(trainer_module, "CometLogger", mock_logger_cls, raising=True)
    return mock_logger_instance


@pytest.fixture
def dummy_data():
    ds = GridDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_weather():
    ds = WeatherDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_multi_aux():
    ds = MultiAuxDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_wind_grid():
    ds = WindGridDataset()
    return DataLoader(ds, batch_size=2)


@pytest.fixture
def dummy_data_wind_and_weather():
    ds = WindAndWeatherDataset()
    return DataLoader(ds, batch_size=2)


def patch_trainer(trainer: Trainer) -> Trainer:
    """Replace loss and metrics with lightweight stubs for isolated testing."""
    trainer.loss_fn = DummyLoss()
    trainer.metric_functions = {"dummy": dummy_metric, "spearman": dummy_metric}
    return trainer


# Tests — baseline (spatial-only) Trainer
def test_trainer_uses_logger(dummy_config_with_logger, mock_comet_logger):
    Trainer(dummy_config_with_logger, spatial_input_channels=SPATIAL_CHANNELS)
    mock_comet_logger.log_params.assert_called()


def test_trainer_setup(dummy_config):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    assert trainer.spatial_input_channels == SPATIAL_CHANNELS
    assert trainer.model is not None
    assert trainer.loss_fn is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert "mse" in trainer.metric_functions


def test_trainer_passes_coordconv_to_model(tmp_path):
    config = _make_config(tmp_path)
    config.model.use_coordconv = True

    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)

    assert trainer.model.use_coordconv is True


def test_trainer_step(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    batch = next(iter(dummy_data))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_trainer_passes_patch_metadata_to_hex_rank_loss(tmp_path):
    config = _make_config(tmp_path)
    config.optimizer.loss = ["kl", "ccc", "hex_mean_pairwise_rank", "hex_top10_pairwise_rank"]
    config.optimizer.loss_weights = {
        "kl": 0.45,
        "ccc": 0.45,
        "hex_mean_pairwise_rank": 0.05,
        "hex_top10_pairwise_rank": 0.05,
    }
    config.data.include_patch_metadata = True
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    batch = next(iter(DataLoader(PatchMetadataDataset(size=2), batch_size=2)))

    _, loss, loss_parts, _, _ = trainer._step(batch)

    assert torch.isfinite(loss)
    assert loss_parts is not None
    assert set(loss_parts) == set(config.optimizer.loss)


def test_trainer_rejects_hex_rank_loss_without_patch_metadata(tmp_path):
    config = _make_config(tmp_path)
    config.optimizer.loss = ["kl", "hex_mean_pairwise_rank"]
    config.optimizer.loss_weights = {"kl": 0.9, "hex_mean_pairwise_rank": 0.1}
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)
    batch = next(iter(DataLoader(GridDataset(size=2), batch_size=2)))

    with pytest.raises(ValueError, match="requires patch metadata"):
        trainer._step(batch)


def test_metric_tensors_inverse_log_standard(tmp_path, monkeypatch):
    mean = 2.0
    std = 0.5
    seen = {}

    def fake_get_output_log_stats(root_dir, output_type):
        seen["log_stats"] = (root_dir, output_type)
        return mean, std

    monkeypatch.setattr("src.trainer.get_output_log_stats", fake_get_output_log_stats)
    config = _make_config(
        tmp_path,
        grid_params=GridParams(
            feature_names_list=["dummy_feat"],
            target_name="fi",
            out_norm="log_standard",
        ),
    )
    trainer = Trainer(config, spatial_input_channels=SPATIAL_CHANNELS)

    predictions = torch.tensor([[[[0.0, 1.0]]]])
    targets = torch.tensor([[[[-1.0, 0.5]]]])
    metric_predictions, metric_targets = trainer._prepare_metric_tensors(predictions, targets)

    expected_predictions = torch.expm1(predictions * std + mean).clamp_min(0.0)
    expected_targets = torch.expm1(targets * std + mean).clamp_min(0.0)
    assert seen["log_stats"] == ("", "fire_intensity")
    assert torch.allclose(metric_predictions, expected_predictions)
    assert torch.allclose(metric_targets, expected_targets)


def test_train_epoch_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_validate_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.validate(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_validate_return_predictions(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4  # dataset size
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_save_and_load_model(tmp_path, dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.train_epoch(dummy_data)
    save_path = trainer.save_model(epoch=1, metric_value=0.5)

    assert tmp_path.joinpath("last.pth").exists()

    checkpoint = trainer.load_model(path=save_path)
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint


def test_run_training(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    trainer.run_training(dummy_data, dummy_data)


def test_test_method(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    results = trainer.test(dummy_data)
    assert "loss" in results
    assert "dummy" in results


# Tests — multi-source Trainer with weather encoder
def test_auxiliary_trainer_setup(auxiliary_config):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_auxiliary_trainer_step(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_weather))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_auxiliary_train_epoch_runs(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_weather)
    assert "loss" in results
    assert "dummy" in results


def test_auxiliary_validate_runs(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_weather)
    assert "loss" in results
    assert "dummy" in results


def test_auxiliary_validate_return_predictions(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data_weather, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_auxiliary_run_training(auxiliary_config, dummy_data_weather):
    trainer = Trainer(
        auxiliary_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_weather, dummy_data_weather)


# Tests — multi-source Trainer with weather + fire_size encoders
def test_multi_aux_trainer_setup(multi_aux_config):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS, "fire_size": FIRE_SIZE_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None


def test_multi_aux_trainer_step(multi_aux_config, dummy_data_multi_aux):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS, "fire_size": FIRE_SIZE_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_multi_aux))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_multi_aux_train_epoch_runs(multi_aux_config, dummy_data_multi_aux):
    trainer = Trainer(
        multi_aux_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"weather": WEATHER_FEATS, "fire_size": FIRE_SIZE_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_multi_aux)
    assert "loss" in results
    assert "dummy" in results


# Tests — multi-source Trainer with WindFeatureEncoder (spatial wind grid)
def test_wind_grid_trainer_setup(wind_grid_config):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_wind_grid_trainer_step(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_wind_grid))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_wind_grid_train_epoch_runs(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_wind_grid)
    assert "loss" in results
    assert "dummy" in results


def test_wind_grid_validate_runs(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_wind_grid)
    assert "loss" in results
    assert "dummy" in results


def test_wind_grid_validate_return_predictions(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data_wind_grid, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4  # dataset size
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range after sigmoid"


def test_wind_grid_run_training(wind_grid_config, dummy_data_wind_grid):
    trainer = Trainer(
        wind_grid_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_wind_grid, dummy_data_wind_grid)


# Tests — multi-source Trainer with WindFeatureEncoder + TabularFeatureEncoder (wind_grid + weather)
def test_wind_and_weather_trainer_setup(wind_and_weather_config):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "weather": WEATHER_FEATS},
    )
    assert trainer.auxiliary is True
    assert trainer.model is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)


def test_wind_and_weather_trainer_step(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    batch = next(iter(dummy_data_wind_and_weather))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_wind_and_weather_train_epoch_runs(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data_wind_and_weather)
    assert "loss" in results
    assert "dummy" in results


def test_wind_and_weather_validate_runs(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    results = trainer.validate(dummy_data_wind_and_weather)
    assert "loss" in results
    assert "dummy" in results


def test_wind_and_weather_run_training(wind_and_weather_config, dummy_data_wind_and_weather):
    trainer = Trainer(
        wind_and_weather_config,
        spatial_input_channels=SPATIAL_CHANNELS,
        auxiliary_input_dims={"wind_grid_mixer": WIND_CHANNELS, "weather": WEATHER_FEATS},
    )
    patch_trainer(trainer)
    trainer.run_training(dummy_data_wind_and_weather, dummy_data_wind_and_weather)
