"""
End-to-end script for evaluation of one hexel
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import yaml

from src.config import Config, GridParams
from src.datasets.dataset import get_test_dataloader
from src.datasets.postprocessing.utils import evaluate_and_visualize_hexels, print_and_log_eval_metrics
from src.datasets.utils import get_dataset_dimensions
from src.trainer import Trainer
from src.utils import (
    seed_everything,
    visualize_model_predictions,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate UNet baseline.")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/default_v1.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--stitch_mode",
        type=str,
        default="mean",
        choices=["mean", "max", "center_crop"],
        help="How to stitch the hexel from overlapping windows: 'mean', 'max', or 'center_crop'.",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=128,
        help="What is the window size of the patch",
    )
    parser.add_argument(
        "--visualize_predictions",
        action="store_true",
        help="Boolean flag to visualize some random predictions vs. targets",
    )
    parser.add_argument(
        "--save_visualizations",
        action="store_true",
        help="Boolean flag to save the visualization figure.",
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

    config.logger.enabled = False

    print("\n[Evaluation] Loading test set...")
    # NOTE: If we need the stats on a particular hexel then modify the test_indices.csv in the config file with
    # meta_hex_{hex_id}.csv file
    start_time = time.time()
    test_loader = get_test_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)

    # Get all data sources from the test dataset
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(test_loader.dataset)
    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Auxiliary={auxiliary_input_dims}")

    trainer = Trainer(config, spatial_input_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims)

    # ---------- Load best checkpoint ----------
    # Try best.pth first, fall back to last.pth if needed
    model_ckpt = None
    try:
        print(f"\n[Checkpoint] Loading {config.evaluation.checkpoint_filename} for evaluation...")
        model_ckpt = trainer.load_model(filename=config.evaluation.checkpoint_filename)
    except (FileNotFoundError, AttributeError):
        raise ValueError("[Checkpoint] checkpoint file not found or invalid...")  # noqa: B904

    if model_ckpt is not None:
        print(f"[Checkpoint] Loaded epoch={model_ckpt.get('epoch', 'N/A')} " f"Checkpoint Metrics={model_ckpt.get('metric_value', 'N/A')}")

    # ---------- Evaluation ----------

    source_map = {s.name: s for s in config.data.input_sources}
    grid_source = source_map.get("grid") if "grid" in source_map else None
    grid_features = None
    out_norm = "min_max"  # default fallback, prevent mypy crash
    if grid_source and isinstance(grid_source.params, GridParams):
        grid_features = grid_source.params.feature_names_list
        out_norm = grid_source.params.out_norm

    preds_start_time = time.time()
    test_metrics, test_predictions = trainer.test(test_loader, return_predictions=True)
    preds_time = time.time() - preds_start_time

    if args.visualize_predictions and isinstance(test_predictions, np.ndarray):
        # get the channel mapping dict if it exists
        json_pattern = os.path.join(config.data.root_dir, "feature_channel_map_*.json")
        json_files = glob.glob(json_pattern)

        channel_map = None
        if json_files:
            with open(json_files[0], "r") as f:
                channel_map = json.load(f)

        # save path for visualization figure (if True)
        viz_save_path = None
        if args.save_visualizations:
            viz_save_path = os.path.join(config.save_dir, "inference_samples_examples.png")

        visualize_model_predictions(
            test_loader=test_loader,
            test_predictions=test_predictions,
            save_path=viz_save_path,
            channel_map=channel_map,
            feature_names_list=grid_features,
        )

    # Save predictions
    np.save(os.path.join(config.save_dir, "test_predictions.npy"), test_predictions)

    if isinstance(test_predictions, np.ndarray):
        hexel_metrics = evaluate_and_visualize_hexels(
            test_predictions=test_predictions,
            config=config,
            out_norm=out_norm,
            device=trainer.device,
            experiment_logger=None,
            metric_functions=trainer.metric_functions,
            stitch_mode=args.stitch_mode,
            window_size=int(args.window_size),
            center_crop_size=int(args.window_size) // 2,
        )

        # print metrics in terminal and log into comet
        print_and_log_eval_metrics(test_metrics=test_metrics, hexel_metrics=hexel_metrics, experiment_logger=trainer.logger)

    print(f"=======Total Evaluation Time {round(time.time()-start_time, 3)}s========")
    print(f"=======Prediction Time {round(preds_time, 3)}s========")


if __name__ == "__main__":
    main()
