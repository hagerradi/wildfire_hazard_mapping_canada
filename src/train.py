"""
End-to-end script for running training and evaluation
"""

import argparse
import os

import yaml

from src.config import Config
from src.datasets.dataset import get_test_dataloader, get_train_val_dataloader
from src.trainer import Trainer
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_v1.yaml",
        help="Path to YAML config file.",
    )
    return parser.parse_args()


def load_config(path: str) -> Config:
    """
    load yaml config
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return Config(**raw)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    # ---------- Set Seed ----------
    seed = getattr(config, "seed", 42)
    deterministic = getattr(config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    # ---------- Data ----------
    train_loader, val_loader = get_train_val_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)

    dataset = train_loader.dataset
    sources = getattr(dataset, "sources", {})

    spatial_channels = None
    tabular_input_dims = {}

    # Get dims. of all sources.
    for name, source in sources.items():
        # Base source: spatial grid.
        if name == "grid":
            spatial_channels = source.input_dim()
        # The extra features (tabular).
        else:
            tabular_input_dims[name] = source.input_dim()

    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Tabular={tabular_input_dims}")

    # ---------- Training ----------
    trainer = Trainer(config=config, spatial_input_channels=spatial_channels, tabular_input_dims=tabular_input_dims)

    trainer.run_training(
        train_loader=train_loader,
        val_loader=val_loader,
    )
    # ---------- Load best checkpoint ----------
    # Try best.pth first, fall back to last.pth if needed
    best_ckpt = None
    try:
        print("\n[Checkpoint] Loading best.pth for evaluation...")
        best_ckpt = trainer.load_model(filename="best.pth")
    except FileNotFoundError:
        print("[Checkpoint] best.pth not found, falling back to last.pth...")
        best_ckpt = trainer.load_model(filename="last.pth")

    if best_ckpt is not None:
        print(f"[Checkpoint] Loaded epoch={best_ckpt.get('epoch', 'N/A')} " f"Checkpoint Metrics={best_ckpt.get('metric_value', 'N/A')}")

    # ---------- Evaluation ----------
    print("\n[Evaluation] Running on test set...")
    test_loader = get_test_dataloader(
        config=config.data,
        modelling_approach=config.modelling_approach,
    )
    test_metrics = trainer.test(test_loader)
    if isinstance(test_metrics, tuple):
        test_metrics = test_metrics[0]

    print("\n[Test metrics]")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}")

    # Log test results to comet, at the end
    if trainer.logger:
        trainer.logger.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})


if __name__ == "__main__":
    main()
