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


def _make_config(
    tmp_path,
    *,
    logger_enabled: bool = False,
    input_feature_list: list[str] | None = None,
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
                    params=GridParams(feature_names_list=["dummy_feat"]),
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
def mock_comet_logger(monkeypatch):
    """
    Mock the CometLogger inside src.trainer so no API key or network call is
    needed. Apply this fixture explicitly to tests that enable the logger.
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


def test_trainer_step(dummy_config, dummy_data):
    trainer = Trainer(dummy_config, spatial_input_channels=SPATIAL_CHANNELS)
    patch_trainer(trainer)
    batch = next(iter(dummy_data))
    preds, loss, loss_parts, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


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
