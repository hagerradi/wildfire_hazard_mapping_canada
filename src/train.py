"""
End-to-end script for running training and evaluation
"""

import argparse
import os

import yaml

from src.config import Config
from src.datasets.dataloader import get_test_loader, get_train_val_dataloader
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

    # ---------- Training ----------
    trainer = Trainer(config)
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
        print(f"[Checkpoint] Loaded epoch={best_ckpt.get('epoch', 'N/A')} " f"loss={best_ckpt.get('loss', 'N/A')}")

    # ---------- Evaluation ----------
    print("\n[Evaluation] Running on test set...")
    test_loader = get_test_loader(
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
