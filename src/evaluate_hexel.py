"""
End-to-end script for evaluation of one hexel
"""

import argparse
import os

import numpy as np
import yaml

from src.config import Config
from src.datasets.dataloader import get_test_loader
from src.trainer import Trainer
from utils import visualize_model_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_v1.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--visualize_predictions",
        type=bool,
        default=True,
        help="Boolean flag to visualize some random predictions vs. targets",
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

    trainer = Trainer(config)

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
    test_metrics, test_predictions = trainer.test(test_loader, return_predictions=True)
    # Save predictions
    np.save(os.path.join(config.save_dir, "test_predictions.npy"), test_predictions)

    if args.visualize_predictions:
        visualize_model_predictions(test_loader, test_predictions)

    print("\n[Test metrics]")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    # TODO: Stitch predictions back to hexel
    # TODO: Merge predictions of multiple scenarios for approach 2
    # TODO: re-compute metrics at the hexel level


if __name__ == "__main__":
    main()
