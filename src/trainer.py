import os
import time
from typing import Any, cast

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.datasets.utils import compute_number_input_channels
from src.logger import CometLogger
from src.losses import WeightedLoss
from src.metrics import compute_bias, compute_mae, compute_mse, compute_spearman, compute_ssim
from src.models.baselines import UNet
from src.models.utils import get_nbr_model_parameters
from src.models.fusion_unet import FusionUNet 
from utils import build_single_loss


class Trainer:
    def __init__(
        self,
        config: Config,
        grid_channel_dim: int,
        weather_input_dim: int = 0,
    ):
        self.config = config
        self.grid_channel_dim = grid_channel_dim
        self.weather_input_dim = weather_input_dim
        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available() and torch.backends.mps.is_built()
            else "cpu"
        )
        print(f"\n[Device] Using: {self.device}")

        self.save_dir = self.config.save_dir
        os.makedirs(self.save_dir, exist_ok=True)

        self.logger = None
        if self.config.logger.enabled:
            self.logger = CometLogger(
                project_name=self.config.logger.project_name,
                workspace=self.config.logger.workspace,
                experiment_name=self.config.logger.experiment_name,
                experiment_tags=self.config.logger.tags,
            )
            self.log_every_n_step = self.config.logger.log_every_n_step
            self.logger.log_params(self.config.model_dump())
        self.setup()

    def setup(self):
        """
        Define model, loss function and optimizer.
        """
        # 1. Model Instantiation
        use_spatial_encoder = getattr(self.config.model, "use_spatial_encoder", True)
        
        if self.grid_channel_dim == 0:
            print("[Trainer] Grid channels detected as 0. Forcefully disabling spatial encoder.")
            use_spatial_encoder = False

        if self.weather_input_dim > 0:
            print(f"[Trainer] FusionUNet instantiated. Grid Chans: {self.grid_channel_dim}, Weather Feats: {self.weather_input_dim}")
            self.model = FusionUNet(
                input_channels=self.grid_channel_dim,
                num_classes=self.config.model.num_classes,
                weather_input_dim=self.weather_input_dim,
                weather_embed_dim=getattr(self.config.model, "weather_embed_dim", 64),
                pooling_type=getattr(self.config.model, "pooling_type", "attention"),
                use_spatial_encoder=use_spatial_encoder, # Pass the corrected flag
            )
        else:
            print(f"[Trainer] Standard UNet instantiated. Grid Chans: {self.grid_channel_dim}")
            self.model = UNet(
                input_channels=self.grid_channel_dim,
                num_classes=self.config.model.num_classes,
            )

        self.model.to(self.device)

        # Log Params
        total_params, trainable_params = get_nbr_model_parameters(self.model)
        print(f"Model Params: Total={total_params:,} | Trainable={trainable_params:,}")
        if self.logger:
            self.logger.log_params({"model_total_params": total_params, "model_trainable_params": trainable_params})

        # 2. Setup Loss
        loss_config = self.config.optimizer.loss
        if isinstance(loss_config, str):
            self.loss_fn = build_single_loss(loss_config)
        else:
            loss_names = loss_config
            weights = self.config.optimizer.loss_weights
            losses = {n: build_single_loss(n) for n in loss_names}
            self.loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

        # 3. Setup Optimizer
        opt_name = self.config.optimizer.name
        opt_params = {"lr": self.config.optimizer.lr}
        OptimizerClass = getattr(optim, opt_name)
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_params)

        self.global_step = 0

        # 4. Metrics
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
                raise ValueError(f"Metric '{name}' not implemented. Options: {list(available_metrics.keys())}")

    def _step(self, batch: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor] | None, torch.Tensor, torch.Tensor]:
        """
        Returns (predictions_probs, total_loss, loss_parts, targets, masks)
        """
        weather = None
        if len(batch) == 4:
            inputs, targets, masks, weather = batch
            weather = weather.to(self.device)
        else:
            inputs, targets, masks = batch

        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        masks = masks.to(self.device)

        # If we expect 0 grid channels, we ignore the inputs
        if self.grid_channel_dim == 0:
            inputs = None
        # Also check for the scalar placeholder from the dataset
        elif inputs.numel() == 1:
            inputs = None

        # Forward Pass
        if isinstance(self.model, FusionUNet):
            logits = self.model(inputs, weather)
        else:
            logits = self.model(inputs)

        loss_out = self.loss_fn(logits, targets, masks)

        if isinstance(loss_out, tuple):
            total_loss, loss_parts = loss_out
        else:
            total_loss = cast(torch.Tensor, loss_out)
            loss_parts = None

        predictions_probs = torch.sigmoid(logits)

        return predictions_probs, total_loss, loss_parts, targets, masks

    def train_epoch(self, loader: DataLoader) -> dict[str, float]:
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

            batch_size = targets.size(0) if hasattr(targets, "size") else 1
            running_loss += loss.item() * batch_size
            running_batch_count += batch_size

            if self.logger and self.global_step % self.log_every_n_step == 0:
                self.logger.log_metrics({"train_step_loss": loss.item()}, step=self.global_step)

            training_loop.set_description(f"Loss: {running_loss / running_batch_count:.4f}")

            with torch.no_grad():
                for name, metric_fn in self.metric_functions.items():
                    value = metric_fn(predictions.detach(), targets, masks)
                    running_metrics[name] += value.item() * batch_size
                    
                    if self.logger and self.global_step % self.log_every_n_step == 0:
                        self.logger.log_metrics({f"train_step_{name}": value.item()}, step=self.global_step)
                        if loss_parts is not None:
                            self.logger.log_metrics(
                                {f"train_step_loss_{k}": v.item() for k, v in loss_parts.items()},
                                step=self.global_step,
                            )
                            
                if loss_parts is not None and running_loss_parts is not None:
                    for k, v in loss_parts.items():
                        running_loss_parts[k] += v.item() * batch_size

            self.global_step += 1

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        
        if running_loss_parts is not None:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

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

            for name, metric_fn in self.metric_functions.items():
                value = metric_fn(predictions.detach(), targets, masks)
                running_metrics[name] += value.item() * batch_size
            
            if loss_parts is not None and running_loss_parts is not None:
                for k, v in loss_parts.items():
                    running_loss_parts[k] += v.item() * batch_size

        avg_loss = running_loss / max(1, running_batch_count)
        results = {"loss": avg_loss}
        
        if running_loss_parts is not None:
            for k, total_v in running_loss_parts.items():
                results[f"loss_{k}"] = total_v / max(1, running_batch_count)

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

    def run_training(self, train_loader: DataLoader, val_loader: DataLoader):
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
                    print(f"Warning: Best metric '{metric_key}' not found. Defaulting to 'loss'.")
                    metric_key = "loss"
                    metric_mode = "min"
                
                if best_val_metric is None or self._is_metric_better(val_result[metric_key], best_val_metric, metric_mode):
                    best_val_metric = val_result[metric_key]
                    if self.save_dir:
                        best_path = self.save_model(epoch=epoch, metric_value=best_val_metric, filename="best.pth")
                        if self.logger:
                            self.logger.experiment.log_model(name="best", file_or_folder=best_path, overwrite=True)

            self.save_model(epoch=epoch, metric_value=val_result.get(metric_key, 0.0) if val_result else 0.0)

    def save_model(self, epoch: int, metric_value: float, filename: str = "last.pth"):
        if not self.save_dir:
            return None
        
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