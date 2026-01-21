import os
import time
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.datasets.utils import compute_number_input_channels
from src.logger import CometLogger
from src.losses import BCELoss, DiceLoss, FocalLoss, MAELoss, MSELoss
from src.metrics import compute_bias, compute_mae, compute_mse, compute_spearman, compute_ssim
from src.models.baselines import UNet
from src.models.utils import get_nbr_model_parameters


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

        self.logger = None
        # Only initialize logger if not in test-only mode
        if self.config.logger.enabled:
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
        # Automatically infer input channels based on data config
        self.input_channels = compute_number_input_channels(
            feature_names_list=self.config.data.feature_names_list,
            fuel_feats_encoding=self.config.data.fuel_feats_encoding,
            root_dir=self.config.data.root_dir,
            modelling_approach=self.config.modelling_approach,
        )
        print(f"[Trainer] Auto-inferred Input Channels: {self.input_channels}")

        self.model = UNet(input_channels=self.input_channels, num_classes=self.config.model.num_classes)
        self.model.to(self.device)

        # Get model nbr of params and log them into Logger
        total_params, trainable_params = get_nbr_model_parameters(self.model)
        print(f"Model Params: Total={total_params:,} | Trainable={trainable_params:,}")
        if self.logger:
            self.logger.log_params({"model_total_params": total_params, "model_trainable_params": trainable_params})

        # setup loss
        loss_name = str(self.config.optimizer.loss_name).lower()

        if loss_name in ["bce", "bceloss"]:
            self.loss_fn = BCELoss()
        elif loss_name in ["mse", "mseloss"]:
            self.loss_fn = MSELoss()
        elif loss_name in ["mae", "maeloss"]:
            self.loss_fn = MAELoss()
        elif loss_name in ["focal", "focalloss"]:
            self.loss_fn = FocalLoss()
        elif loss_name in ["dice", "diceloss"]:
            self.loss_fn = DiceLoss()
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
            "bias": compute_bias,
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
        if self.config.optimizer.loss_name in ["bce", "bceloss", "focal", "focalloss"]:
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

            if self.logger and self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            # compute the metrics
            with torch.no_grad():
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(predictions.detach(), targets, masks)
                    running_metrics[name] += value.item() * batch_size
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        return results

    @torch.no_grad()
    def validate(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
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
    def test(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        return self.validate(loader, return_predictions=return_predictions)

    @staticmethod
    def _is_metric_better(curr, best, best_mode):
        if best is None:
            return True
        return curr > best if best_mode == "max" else curr < best

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        num_epochs = self.config.training.max_epochs
        log_every_n_epoch = self.config.training.log_every_n_epoch

        best_val_metric = None
        metric_key = getattr(getattr(self.config, "evaluation", None), "best_ckpt_metric", "spearman")
        metric_mode = (getattr(self.config, "best_ckpt_metric_mode", None) or "max").lower()

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

                if self.logger:
                    self.logger.log_metrics(metrics_to_log, epoch=epoch)

            if val_result is not None:
                if metric_key not in val_result:
                    raise KeyError(
                        f"Best metric '{metric_key}' not found in val_result keys={list(val_result.keys())}. "
                        f"Either compute it in validation or change config.evaluation.best_metric."
                    )
                if metric_mode not in ("max", "min"):
                    raise ValueError(f"evaluation.best_mode must be 'max' or 'min', got: {metric_mode}")

                if best_val_metric is None or self._is_metric_better(val_result[metric_key], best_val_metric, metric_mode):
                    best_val_metric = val_result[metric_key]
                    # auto-save best if save_dir configured
                    if self.save_dir:
                        best_path = self.save_model(epoch=epoch, metric_value=best_val_metric, filename="best.pth")

                        # log best model to comet
                        if self.logger:
                            self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

            # save most recent checkpoint
            self.save_model(epoch=epoch, metric_value=val_result[metric_key])

    def save_model(self, epoch: int, metric_value: float, filename: str = "last.pth"):
        if not self.save_dir:
            raise ValueError("save_dir not set")

        path = os.path.join(self.save_dir, filename)

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric_value": metric_value,
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
