"""
End-to-end script for running training and evaluation
"""

import argparse
import os

import numpy as np
import yaml

from src.config import Config, GridParams
from src.datasets.dataset import get_test_dataloader, get_train_val_dataloader
from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels, print_and_log_eval_metrics
from src.datasets.utils import get_dataset_dimensions
from src.trainer import Trainer
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/bp_common_input_pipeline.yaml",
        help="Path to YAML config file.",
    )
    # logging is enabled by default, unless you pass --no_log_test_predicted_hexels
    parser.add_argument(
        "--no_log_test_predicted_hexels",
        dest="log_test_predicted_hexels",
        action="store_false",
        default=True,
        help="Disable saving predicted hexels to comet (default: True)",
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

    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(train_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Auxiliary={auxiliary_input_dims}")

    # ---------- Training ----------
    trainer = Trainer(
        config=config,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims=auxiliary_input_dims,
        train_dataset=train_loader.dataset,
    )

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
        try:
            best_ckpt = trainer.load_model(filename="last.pth")
        except FileNotFoundError:
            print("[Checkpoint] No checkpoint found (last.pth missing)...")
            return

    if best_ckpt is not None:
        print(f"[Checkpoint] Loaded epoch={best_ckpt.get('epoch', 'N/A')} Checkpoint Metrics={best_ckpt.get('metric_value', 'N/A')}")

    # ---------- Evaluation ----------
    print("\n[Evaluation] Running on test set...")
    test_loader = get_test_dataloader(
        config=config.data,
        modelling_approach=config.modelling_approach,
        seed=seed,
    )
    test_metrics, test_predictions = trainer.test(test_loader, return_predictions=True)

    hexel_metrics = {}

    if args.log_test_predicted_hexels:
        source_map = {s.name: s for s in config.data.input_sources}
        grid_source = source_map.get("grid") if "grid" in source_map else None
        out_norm = "min_max"  # default fallback, prevent mypy crash
        if grid_source and isinstance(grid_source.params, GridParams):
            out_norm = grid_source.params.out_norm

        if isinstance(test_predictions, np.ndarray):  # for mypy
            hexel_metrics = evaluate_and_visualize_hexels(
                test_predictions=test_predictions,
                config=config,
                out_norm=out_norm,
                device=trainer.device,
                experiment_logger=trainer.logger,
                metric_functions=trainer.metric_functions,
            )

    # print metrics in terminal and log into comet
    print_and_log_eval_metrics(test_metrics=test_metrics, hexel_metrics=hexel_metrics, experiment_logger=trainer.logger)


if __name__ == "__main__":
    main()
