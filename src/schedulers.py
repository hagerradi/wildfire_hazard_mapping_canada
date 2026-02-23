from typing import Any

import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import Config


def build_scheduler(config: Config, optimizer: optim.Optimizer, train_loader: DataLoader) -> tuple[Any, str | None]:
    """
    Factory function to build a learning rate scheduler.

    Args:
        config (Config): the config object.
        optimizer (optim.Optimizer): an optimizer object.
        train_loader (DataLoader): dataloader object.

    Returns: (scheduler_instance, step_frequency_type): tuple with scheduler object and type.
    """
    # base setup without scheduler
    if not config.scheduler.name:
        return None, None

    name = config.scheduler.name.lower()
    epochs = config.training.max_epochs
    steps_per_epoch = len(train_loader)
    total_steps = epochs * steps_per_epoch

    # onecycle
    if name == "onecycle":
        return optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.scheduler.max_lr,
            total_steps=total_steps,
        ), "batch"

    # cosine warmup
    elif name == "cosine_warmup":
        warmup_steps = max(1, config.scheduler.warmup_epochs * steps_per_epoch)
        warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))

        return optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps]
        ), "batch"

    # plateau
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=config.scheduler.factor, patience=config.scheduler.patience
        ), "epoch_metric"

    # multistep
    elif name == "multistep":
        return optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.scheduler.milestones, gamma=config.scheduler.factor), "epoch"

    else:
        raise ValueError(f"Unknown scheduler: {name}")
