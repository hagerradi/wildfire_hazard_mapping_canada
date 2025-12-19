import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import Config
from src.trainer import Trainer


class DummyLogger:
    def __init__(self, *args, **kwargs):
        pass

    def log_params(self, params):
        pass

    def log_metrics(self, metrics, step=None, epoch=None):
        pass

    class Experiment:
        def log_model(self, name, file_or_folder, overwrite):
            pass

    experiment = Experiment()


class DummyLoss(torch.nn.Module):
    def forward(self, predictions, targets, masks):
        return torch.tensor(1.0, requires_grad=True)


def dummy_metric(predictions, targets, masks):
    return torch.tensor(0.5)


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
            "input_channels": 1,
            "num_classes": 1,
        },
        "optimizer": {
            "loss_name": "mse",
            "name": "Adam",
            "lr": 0.001,
        },
        "data": {"root_dir": "", "train_split": "", "val_split": "", "test_split": "", "fuel_feats_encoding": ""},
        "metrics": ["mse"],
        "training": {
            "max_epochs": 1,
            "log_every_n_epoch": 1,
        },
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


def patch_trainer(trainer):
    # Patch logger, loss, metrics for isolated testing
    trainer.logger = DummyLogger()
    trainer.loss_fn = DummyLoss()
    trainer.metric_functions = {"dummy": dummy_metric}
    return trainer


def test_trainer_setup(dummy_config):
    trainer = Trainer(dummy_config)
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


def test_save_and_load_model(tmp_path, dummy_config, dummy_data):
    trainer = Trainer(dummy_config)
    patch_trainer(trainer)
    trainer.train_epoch(dummy_data)
    save_path = trainer.save_model(epoch=1, loss=0.5)
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
