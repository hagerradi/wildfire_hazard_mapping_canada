from unittest.mock import MagicMock

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.trainer import Trainer


class DummyLoss(torch.nn.Module):
    def forward(self, predictions, targets, masks):
        # Constant loss just to exercise the training loop
        return torch.tensor(1.0, requires_grad=True)


def dummy_metric(predictions, targets, masks):
    # Constant metric just to exercise metric logging
    return torch.tensor(0.5)


@pytest.fixture(autouse=True)
def mock_comet_logger(monkeypatch):
    """
    Automatically mock the CometLogger used inside Trainer so that
    no COMET_API_KEY is needed and no external calls are made.
    """
    import src.trainer as trainer_module

    # Class that Trainer will call as CometLogger(...)
    mock_logger_cls = MagicMock(name="CometLogger")

    # Instance returned by that call
    mock_logger_instance = MagicMock(name="logger_instance")
    mock_logger_cls.return_value = mock_logger_instance

    # Patch the CometLogger symbol inside src.trainer
    monkeypatch.setattr(trainer_module, "CometLogger", mock_logger_cls, raising=True)

    # If you want to assert on it later, you can return it
    return mock_logger_instance


def test_trainer_uses_logger(dummy_config, mock_comet_logger):
    Trainer(dummy_config)
    mock_comet_logger.log_params.assert_called()


@pytest.fixture(autouse=True)
def mock_compute_channels(monkeypatch):
    """
    Mocks the channel computation function.
    Instead of reading a JSON file, it simply returns 1 (to match dummy_data).
    """
    import src.trainer as trainer_module

    monkeypatch.setattr(
        trainer_module,
        "compute_number_input_channels",
        lambda **kwargs: 1,  # Always return 1 channel for tests
    )


@pytest.fixture
def dummy_config(tmp_path):
    # Minimal config for Trainer
    config_dict = {
        "save_dir": str(tmp_path),
        "logger": {
            "project_name": "test",
            "workspace": "test",
            "experiment_name": "test",
            "tags": [],
            "log_every_n_step": 1,
        },
        "model": {
            "num_classes": 1,
        },
        "optimizer": {
            "loss_name": "mse",
            "name": "Adam",
            "lr": 0.001,
        },
        "data": {
            "root_dir": "",
            "raw_data_dir": "",
            "train_split": "",
            "val_split": "",
            "test_split": "",
            "feature_names_list": ["dummy_feat"],  # Needs to exist for Trainer init access
            "fuel_feats_encoding": "",
        },
        "metrics": ["mse"],
        "training": {
            "max_epochs": 1,
            "log_every_n_epoch": 1,
        },
        "evaluation": {"best_ckpt_metric": "spearman", "checkpoint_filename": "best.pth"},
        # keep/remove depending on how your Config is defined
        "model_dump": lambda: {},
    }
    return Config(**config_dict)


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    data = torch.rand(4, 1, 32, 32)
    targets = data * 0.9
    masks = torch.rand(4, 1, 32, 32) > 0.5
    ds = TensorDataset(data, targets, masks)
    loader = DataLoader(ds, batch_size=2)
    return loader


def patch_trainer(trainer: Trainer) -> Trainer:
    """
    Patch loss and metric functions for isolated testing.
    Logger is already mocked globally by mock_comet_logger.
    """
    trainer.loss_fn = DummyLoss()
    trainer.metric_functions = {"dummy": dummy_metric, "spearman": dummy_metric}
    return trainer


def test_trainer_setup(dummy_config):
    trainer = Trainer(dummy_config)
    assert trainer.input_channels == 1
    assert trainer.model is not None
    assert trainer.loss_fn is not None
    assert isinstance(trainer.optimizer, torch.optim.Optimizer)
    assert "mse" in trainer.metric_functions


def test_trainer_step(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    batch = next(iter(dummy_data))
    preds, loss, targets, masks = trainer._step(batch)
    assert preds.shape == targets.shape
    assert isinstance(loss, torch.Tensor)


def test_train_epoch_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    results = trainer.train_epoch(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_validate_runs(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    results = trainer.validate(dummy_data)
    assert "loss" in results
    assert "dummy" in results


def test_validate_return_predictions(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    results, preds = trainer.validate(dummy_data, return_predictions=True)
    assert "loss" in results
    assert preds.shape[0] == 4  # batch size * num batches
    assert np.all((preds >= 0) & (preds <= 1)), "Predictions should be in [0, 1] range"


def test_save_and_load_model(tmp_path, dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    trainer.train_epoch(dummy_data)
    save_path = trainer.save_model(epoch=1, metric_value=0.5)

    # Model should be saved into config.save_dir (tmp_path)
    assert tmp_path.joinpath("last.pth").exists()

    checkpoint = trainer.load_model(path=save_path)
    assert "model_state" in checkpoint
    assert "optimizer_state" in checkpoint


def test_run_training(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    trainer.run_training(dummy_data, dummy_data)
    # Should complete without error


def test_test_method(dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    results = trainer.test(dummy_data)
    assert "loss" in results
    assert "dummy" in results
