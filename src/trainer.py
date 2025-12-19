import os
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.logger import CometLogger
from src.losses import BCELoss, MSELoss
from src.metrics import compute_mae, compute_mse, compute_spearman, compute_ssim
from src.models.baselines import UNet


class Trainer:
    def __init__(
        self,
        config: Config,
    ):
        self.config = config

        self.device = (
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu"
        )
        print(f"\n[Device] Using: {self.device}")

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # setup of the logger
        self.logger = CometLogger(
            project_name=self.config.logger.project_name,
            workspace=self.config.logger.workspace,
            experiment_name=self.config.logger.experiment_name,
            experiment_tags=self.config.logger.tags,
        )
        self.log_every_n_step = self.config.logger.log_every_n_step

        # log all the params.
        self.logger.log_params(self.config.model_dump())

        self.setup()

    def setup(self):
        """
        define model, loss function and optimizer.
        """
        self.model = UNet(input_channels=self.config.model.input_channels, num_classes=self.config.model.num_classes)
        self.model.to(self.device)
        # setup loss
        loss_name = str(self.config.optimizer.loss_name).lower()

        if loss_name in ["bce", "bceloss"]:
            self.loss_fn = BCELoss()
        elif loss_name in ["mse", "mseloss"]:
            self.loss_fn = MSELoss()
        else:
            raise ValueError(f"Unknown loss type in config.loss: {self.config.loss}")

        # setup optimizer
        opt_name = self.config.optimizer.name
        # TODO: add other parameters
        opt_params = {
            "lr": self.config.optimizer.lr,
        }

        OptimizerClass = getattr(optim, opt_name)
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_params)

        self.global_step = 0

        # get metrics to compute
        available_metrics = {
            "mse": compute_mse,
            "mae": compute_mae,
            "spearman": compute_spearman,
            "ssim": compute_ssim,
        }

        self.metric_functions = {}

        for name in self.config.metrics:
            if name in available_metrics:
                self.metric_functions[name] = available_metrics[name]
            else:
                raise ValueError(f"Metric '{name}' in config. is not implemented." f"Available options: {list(available_metrics.keys())}")

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Default step. Expects batch -> (inputs, targets, masks).
        Returns (predictions, loss, targets_on_device, masks_on_device).
        """
        inputs, targets, masks = batch
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        masks = masks.to(self.device)

        predictions = self.model(inputs)
        # for bce, we will apply sigmoid after the loss
        if self.config.optimizer.loss_name in ["bce", "bceloss"]:
            loss = self.loss_fn(predictions, targets, masks)
            predictions = torch.sigmoid(predictions)
        else:
            predictions = torch.sigmoid(predictions)
            loss = self.loss_fn(predictions, targets, masks)
        return predictions, loss, targets, masks

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self.metric_functions}

        training_loop = tqdm(loader, desc="Training", leave=True)

        for batch in training_loop:
            predictions, loss, targets, masks = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            if self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            # compute the metrics
            with torch.no_grad():
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(predictions.detach(), targets, masks)
                    running_metrics[name] += value.item() * batch_size
                    if self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        return results

    @torch.no_grad()
    def validate(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], Any]:
        self.model.eval()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self.metric_functions}

        preds_list = []

        for batch in loader:
            predictions, loss, targets, masks = self._step(batch)

            if return_predictions:
                preds_list.append(predictions.detach().cpu().numpy())

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            # compute the metrics
            with torch.no_grad():
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(predictions.detach(), targets, masks)
                    running_metrics[name] += value.item() * batch_size

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        if return_predictions:
            return results, np.concatenate(preds_list, axis=0)

        return results

    @torch.no_grad()
    def test(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], Any]:
        return self.validate(loader, return_predictions=return_predictions)

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        num_epochs = self.config.training.max_epochs
        log_every_n_epoch = self.config.training.log_every_n_epoch
        best_val_loss = None

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_res = self.train_epoch(train_loader)
            elapsed = time.time() - start

            val_result = self.validate(val_loader) if val_loader is not None else None
            if isinstance(val_result, tuple):
                val_result = val_result[0]
            # log metrics and loss
            if epoch % log_every_n_epoch == 0:
                msg = f"Epoch {epoch}/{num_epochs} - train_loss: {train_res['loss']:.4f}"
                if val_result is not None:
                    msg += f", val_loss: {val_result['loss']:.4f}"
                msg += f", time: {elapsed:.1f}s"
                print(msg)

                metrics_to_log = {f"train_{k}": v for k, v in train_res.items()}
                metrics_to_log["epoch_duration"] = elapsed

                if val_result:
                    metrics_to_log.update({f"val_{k}": v for k, v in val_result.items()})

                self.logger.log_metrics(metrics_to_log, epoch=epoch)

            if val_result is not None and (best_val_loss is None or val_result["loss"] < best_val_loss):
                best_val_loss = val_result["loss"]
                # auto-save best if save_dir configured
                if self.save_dir:
                    best_path = self.save_model(epoch=epoch, loss=val_result["loss"], filename="best.pth")

                    # log best model to comet
                    self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

            # save most recent checkpoint
            self.save_model(epoch=epoch, loss=val_result["loss"])

    def save_model(self, epoch: int, loss: float, filename: str = "last.pth"):
        if not self.save_dir:
            raise ValueError("save_dir not set")

        path = os.path.join(self.save_dir, filename)

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
        }
        torch.save(payload, path)
        return path

    def load_model(self, path: str | None = None, filename: str = "last.pth", map_location: str | None = None):
        if path is None:
            path = os.path.join(self.save_dir, filename)

        map_location = map_location or self.device
        checkpoint = torch.load(path, map_location=map_location)

        self.model.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        return checkpoint


# TODO: convert to unit test
if __name__ == "__main__":
    from torch.utils.data import DataLoader, TensorDataset

    # toy dataset
    x = torch.randn(2, 20, 64, 64)
    y = torch.randn(2, 1, 64, 64)
    mask = torch.rand_like(y) > 0.5

    ds = TensorDataset(x, y, mask)
    train_dl = DataLoader(ds, batch_size=32, shuffle=True)
    val_dl = DataLoader(ds, batch_size=64)
    test_dl = DataLoader(ds, batch_size=64)

    with open("configs/default.yaml") as f:
        raw = yaml.safe_load(f)

    config = Config(**raw)
    trainer = Trainer(config)

    trainer.run_training(train_dl, val_loader=val_dl)
    trainer.load_model()
    print("Test:", trainer.test(test_dl))
