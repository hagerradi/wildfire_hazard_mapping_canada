import os
import time
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler, ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.spatial.utils import get_output_log_stats, get_range_output
from src.config import Config, GridParams
from src.datasets.targets import get_target_spec
from src.datasets.utils import denormalize_output_target
from src.logger import CometLogger
from src.losses import WeightedLoss
from src.models.unet import BaselineUNet, MultiSourceUNet
from src.models.utils import get_nbr_model_parameters
from src.schedulers import build_lr_scheduler
from utils import AVAILABLE_METRICS, build_single_loss, set_device


class Trainer:
    def __init__(self, config: Config, spatial_input_channels: int | None = None, auxiliary_input_dims: dict[str, int] | None = None):
        self.config = config
        self.spatial_input_channels = spatial_input_channels
        self.auxiliary_input_dims = auxiliary_input_dims if auxiliary_input_dims is not None else {}

        # Set device
        self.device = set_device()
        print(f"\n[Device] Using: {self.device}")

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        # Comet Logger
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

        # metrics for best checkpoint saving
        self.best_ckpt_metrics = list(self.config.evaluation.best_ckpt_metrics)
        self.best_ckpt_modes = list(self.config.evaluation.best_ckpt_metrics_mode)
        if len(self.best_ckpt_metrics) != len(self.best_ckpt_modes):
            raise ValueError("Number of best_ckpt_metric and best_ckpt_metric_mode must match!")
        self._best_metric_list: list[float] = []

    def setup(self):
        """
        Define model, loss function and optimizer.
        """

        self._grid_params = self._get_grid_params()
        self._target_spec = get_target_spec(self._grid_params.target_name) if self._grid_params is not None else get_target_spec("bp")

        # Flag to indicate we are including auxiliary features
        self.auxiliary = "auxiliary" in self.config.model.input_feature_list

        # Multi-source path: spatial grids + auxiliary data.
        if self.auxiliary:
            if not self.auxiliary_input_dims:
                raise ValueError("Config requests auxiliary features, but no auxiliary dim. were detected.")

            print("[Trainer] Mode: Multi-Source (Spatial + auxiliary)")
            print(f"[Trainer] Spatial Channels: {self.spatial_input_channels}, Auxiliary Dim: {self.auxiliary_input_dims}")

            self.model = MultiSourceUNet(
                input_channels=self.spatial_input_channels,
                num_classes=self.config.model.num_classes,
                hidden_features=self.config.model.hidden_features,
                input_feature_list=self.config.model.input_feature_list,
                use_skip_connections=self.config.model.use_skip_connections,
                use_transpose_conv=self.config.model.use_transpose_conv,
                use_activation_after_upsampling=self.config.model.use_activation_after_upsampling,
                use_coordconv=self.config.model.use_coordconv,
                auxiliary_input_dims=self.auxiliary_input_dims,
                auxiliary_hidden_dims=self.config.model.auxiliary_hidden_dims,
                auxiliary_embed_dims=self.config.model.auxiliary_embed_dims,
                auxiliary_feature_encoder_poolings=self.config.model.auxiliary_feature_encoder_poolings,
            )
        # Single-source path: spatial grids only.
        else:
            print("[Trainer] Mode: Baseline (Spatial Only)")
            print(f"[Trainer] Spatial Channels: {self.spatial_input_channels}")

            self.model = BaselineUNet(
                input_channels=self.spatial_input_channels,
                num_classes=self.config.model.num_classes,
                hidden_features=self.config.model.hidden_features,
                input_feature_list=self.config.model.input_feature_list,
                use_skip_connections=self.config.model.use_skip_connections,
                use_transpose_conv=self.config.model.use_transpose_conv,
                use_activation_after_upsampling=self.config.model.use_activation_after_upsampling,
                use_coordconv=self.config.model.use_coordconv,
            )

        self.model.to(self.device)

        # Get and log number of model params.
        total_params, trainable_params = get_nbr_model_parameters(self.model)
        print(f"Model Params: Total={total_params:,} | Trainable={trainable_params:,}")
        if self.logger:
            self.logger.log_params({"model_total_params": total_params, "model_trainable_params": trainable_params})

        # Setup loss
        loss_config = self.config.optimizer.loss
        huber_beta = self.config.optimizer.huber_beta
        if isinstance(loss_config, str):  # loss is a string
            self.loss_fn = build_single_loss(loss_config, huber_beta=huber_beta)
        else:  # loss is a list
            loss_names = loss_config
            weights = self.config.optimizer.loss_weights
            losses = {n: build_single_loss(n, huber_beta=huber_beta) for n in loss_names}
            self.loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

        # Setup optimizer
        opt_name = self.config.optimizer.name
        # TODO: add other parameters
        opt_params = {
            "lr": self.config.optimizer.lr,
        }

        OptimizerClass = getattr(optim, opt_name)
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_params)

        self.global_step = 0

        # Validate and load metrics from config.
        self._validate_and_load_metrics()

        self._use_sigmoid_predictions = self._target_spec.probability_scale
        self._configure_metric_target_transform()

    def _get_grid_params(self) -> GridParams | None:
        for source in self.config.data.input_sources:
            if source.name == "grid" and isinstance(source.params, GridParams):
                return source.params
        return None

    def _configure_metric_target_transform(self) -> None:
        self._metric_out_norm = "none"
        self._metric_target_min = 0.0
        self._metric_target_max = 1.0
        self._metric_target_log_mean: float | None = None
        self._metric_target_log_std: float | None = None

        if self._grid_params is None:
            return

        self._metric_out_norm = self._grid_params.out_norm
        if self._metric_out_norm == "min_max":
            if self.config.data.raw_data_dir:
                self._metric_target_max, self._metric_target_min = get_range_output(
                    root_dir=self.config.data.raw_data_dir,
                    output_type=self._target_spec.output_type,
                )
        elif self._metric_out_norm == "log_standard":
            self._metric_target_log_mean = self._grid_params.target_log_mean
            self._metric_target_log_std = self._grid_params.target_log_std
            if self._metric_target_log_mean is None or self._metric_target_log_std is None:
                self._metric_target_log_mean, self._metric_target_log_std = get_output_log_stats(
                    root_dir=self.config.data.raw_data_dir,
                    output_type=self._target_spec.output_type,
                )
            if self._metric_target_log_std <= 0.0:
                raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {self._metric_target_log_std}.")
        elif self._metric_out_norm not in {"log", "none", "total_iters", "season_cause_iters"}:
            raise ValueError(f"Unsupported output normalization: {self._metric_out_norm!r}")

    def _prepare_metric_tensors(self, predictions: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self._metric_out_norm == "log_standard" and (self._metric_target_log_mean is None or self._metric_target_log_std is None):
            raise RuntimeError("log_standard metric transform was not configured.")

        metric_predictions = cast(
            torch.Tensor,
            denormalize_output_target(
                data=predictions,
                target_min=self._metric_target_min,
                target_max=self._metric_target_max,
                out_norm=self._metric_out_norm,
                target_log_mean=self._metric_target_log_mean,
                target_log_std=self._metric_target_log_std,
            ),
        )
        metric_targets = cast(
            torch.Tensor,
            denormalize_output_target(
                data=targets,
                target_min=self._metric_target_min,
                target_max=self._metric_target_max,
                out_norm=self._metric_out_norm,
                target_log_mean=self._metric_target_log_mean,
                target_log_std=self._metric_target_log_std,
            ),
        )
        return metric_predictions, metric_targets

    def _validate_and_load_metrics(self) -> None:
        """Helper to validate and load metrics to be computed."""
        if not set(self.config.metrics).issubset(AVAILABLE_METRICS):
            raise ValueError(f"Invalid metrics found.Available options: {list(AVAILABLE_METRICS)}")
        self.metric_functions = {k: AVAILABLE_METRICS[k] for k in self.config.metrics}

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """
        Default step. Expects batch -> {'grid': (inputs, targets, masks), 'weather': ...}.
        Returns (predictions, loss, loss_parts, targets_on_device, masks_on_device).
        """
        # Get the spatial grid inputs, targets and masks.
        if "grid" not in batch:
            raise ValueError("Batch is missing required 'grid' data.")
        inputs, targets, masks = [t.to(self.device) for t in batch["grid"]]

        # Unpack all potential auxiliary data
        auxiliary_data = {}
        for key, value in batch.items():
            if key == "grid":
                continue
            auxiliary_data[key] = value.to(self.device)

        predictions = self.model(inputs, auxiliary_data)

        loss_out = self.loss_fn(predictions, targets, masks)

        # Support if it is a single loss or weighted loss
        if isinstance(loss_out, tuple):
            total_loss, loss_parts = loss_out
        else:
            total_loss = cast(torch.Tensor, loss_out)
            loss_parts = None

        metric_predictions = torch.sigmoid(predictions) if self._use_sigmoid_predictions else predictions
        return metric_predictions, total_loss, loss_parts, targets, masks

    @staticmethod
    def _are_metrics_better(curr: list[float], best: list[float], modes: list[str]):
        if not best:
            return True

        improved = False
        for c, b, mode in zip(curr, best, modes, strict=False):
            if mode == "min":
                if c > b:
                    return False  # a metric got worse!
                elif c < b:
                    improved = True
            elif mode == "max":
                if c < b:
                    return False  # a metric got worse!
                elif c > b:
                    improved = True
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return improved  # Only True if at least one metric improved, none worse

    # Supports any LRScheduler object and metric-based ReduceLROnPlateau schedulers
    def train_epoch(
        self, loader: DataLoader, lr_scheduler: LRScheduler | ReduceLROnPlateau | None = None, lr_scheduler_type: str | None = None
    ) -> dict[str, float]:
        self.model.train()
        running_loss = 0.0
        running_batch_count = 0
        running_metrics = {name: 0.0 for name in self.metric_functions}
        running_loss_parts = None
        if isinstance(self.config.optimizer.loss, list):
            running_loss_parts = {name: 0.0 for name in self.config.optimizer.loss}

        training_loop = tqdm(loader, desc="Training", leave=True)

        for batch in training_loop:
            predictions, loss, loss_parts, targets, masks = self._step(batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # use scheduler if its type is batch-level
            if lr_scheduler is not None and lr_scheduler_type == "batch":
                lr_scheduler.step()

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            if self.logger and self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

                # log LR since it can change with scheduler
                current_lr = self.optimizer.param_groups[0]["lr"]
                self.logger.log_metrics({"learning_rate": current_lr}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(metric_predictions, metric_targets, masks)
                    running_metrics[name] += value.item() * batch_size
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)

                if loss_parts is not None:
                    # log each loss part (raw/unweighted)
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics(
                            {f"train_step_loss_{k}": v.item() for k, v in loss_parts.items()},
                            step=self.global_step,
                        )
                    # accumulate epoch averages
                    if running_loss_parts is not None:
                        for k, v in loss_parts.items():
                            running_loss_parts[k] += v.item() * batch_size

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts is not None:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

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
        running_loss_parts = None
        if isinstance(self.config.optimizer.loss, list):
            running_loss_parts = {name: 0.0 for name in self.config.optimizer.loss}

        preds_list = []
        validation_loop = tqdm(loader, desc="Evaluating", leave=True)

        for batch in validation_loop:
            predictions, loss, loss_parts, targets, masks = self._step(batch)

            if return_predictions:
                preds_list.append(predictions.detach().cpu().numpy())

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            # compute the metrics
            with torch.no_grad():
                metric_predictions, metric_targets = self._prepare_metric_tensors(predictions.detach(), targets)
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(metric_predictions, metric_targets, masks)
                    running_metrics[name] += value.item() * batch_size

                if loss_parts is not None and running_loss_parts is not None:
                    for k, v in loss_parts.items():
                        running_loss_parts[k] += v.item() * batch_size

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        if running_loss_parts is not None:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

        # add averaged metrics to results
        for name, total_value in running_metrics.items():
            results[name] = total_value / max(1, running_batch_count)

        if return_predictions:
            return results, np.concatenate(preds_list, axis=0)

        return results

    @torch.no_grad()
    def test(self, loader: DataLoader, return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray]:
        return self.validate(loader, return_predictions=return_predictions)

    def run_training(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ):
        num_epochs = self.config.training.max_epochs
        log_every_n_epoch = self.config.training.log_every_n_epoch

        # get scheduler and its type
        lr_scheduler, lr_scheduler_type = build_lr_scheduler(self.config, self.optimizer, train_loader)

        for epoch in range(1, num_epochs + 1):
            start = time.time()
            train_res = self.train_epoch(train_loader, lr_scheduler=lr_scheduler, lr_scheduler_type=lr_scheduler_type)
            elapsed = time.time() - start

            val_result = self.validate(val_loader) if val_loader is not None else None
            if isinstance(val_result, tuple):
                val_result = val_result[0]

            # for epoch level schedulers
            if lr_scheduler is not None:
                if lr_scheduler_type == "epoch":
                    lr_scheduler.step()
                elif lr_scheduler_type == "epoch_metric":
                    # Plateau needs a metric to watch. Default to val_loss, fallback to train_loss
                    watch_metric = val_result["loss"] if val_result else train_res["loss"]
                    lr_scheduler.step(watch_metric)

            # log metrics and loss
            if epoch % log_every_n_epoch == 0:
                current_lr = self.optimizer.param_groups[0]["lr"]
                msg = f"Epoch {epoch}/{num_epochs} - train_loss: {train_res['loss']:.4f}"
                if val_result is not None:
                    msg += f", val_loss: {val_result['loss']:.4f}"
                msg += f", lr: {current_lr:.2e}, time: {elapsed:.1f}s"
                print(msg)

                metrics_to_log = {f"train_{k}": v for k, v in train_res.items()}
                metrics_to_log["epoch_duration"] = elapsed

                if val_result:
                    metrics_to_log.update({f"val_{k}": v for k, v in val_result.items()})

                if self.logger:
                    self.logger.log_metrics(metrics_to_log, epoch=epoch)

            # Save best checkpoint based on multi-metrics
            if val_result:
                curr_metric_list = [val_result[m] for m in self.best_ckpt_metrics]
                if self._are_metrics_better(curr_metric_list, self._best_metric_list, self.best_ckpt_modes):
                    self._best_metric_list = curr_metric_list
                    if self.save_dir:
                        best_path = self.save_model(
                            epoch=epoch,
                            metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)},
                            filename="best.pth",
                        )
                        if self.logger:
                            self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

                # save most recent checkpoint
                self.save_model(epoch=epoch, metric_value={m: v for m, v in zip(self.best_ckpt_metrics, curr_metric_list, strict=False)})

    def save_model(self, epoch: int, metric_value: float | dict, filename: str = "last.pth"):
        if not self.save_dir:
            raise ValueError("save_dir not set")

        path = os.path.join(self.save_dir, filename)

        payload = {
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "epoch": epoch,
            "metric_value": metric_value,
            "config": self.config.model_dump(),
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
