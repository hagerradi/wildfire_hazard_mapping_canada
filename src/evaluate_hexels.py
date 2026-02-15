"""
End-to-end script for evaluation of one hexel
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import yaml

from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.grid_loader.utils import get_range_burn_count, get_range_burn_prob
from data_preparation.utils import find_simulation_output_file
from src.config import Config, GridParams
from src.datasets.dataset import get_test_dataloader
from src.datasets.postprocessing.utils import get_predicted_hexel, save_predicted_hexels
from src.datasets.postprocessing.visualize_predictions import visualize_burn_prob_grid
from src.trainer import Trainer
from src.utils import seed_everything, visualize_model_predictions


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
    dataset = test_loader.dataset
    sources = getattr(dataset, "sources", {})
    spatial_channels = None
    aux_input_dims = {}

    for name, source in sources.items():
        if name == "grid":
            spatial_channels = source.input_dim()
        else:
            aux_input_dims[name] = source.input_dim()

    print(f"Detected Data Dimensions: Spatial={spatial_channels} | Aux={aux_input_dims}")

    trainer = Trainer(config, spatial_input_channels=spatial_channels, tabular_input_dims=aux_input_dims)

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

    source_map = {s.name: s for s in config.data.sources}
    grid_source = source_map.get("grid") if "grid" in source_map.keys() else None
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

    print("\n[Test metrics]")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    modelling_approach = config.modelling_approach
    valid_mask_threshold = config.data.valid_mask_threshold
    output_type, season, cause = "prob", None, None
    if modelling_approach == "1":
        max_target_val, min_target_val = get_range_burn_prob(root_dir=raw_data_dir)
    else:
        max_target_val, min_target_val = get_range_burn_count(root_dir=raw_data_dir)

    if isinstance(test_predictions, str):
        # Handle the error or raise an exception
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    try:
        test_df = pd.read_csv(os.path.join(data_dir, config.data.test_split))
    except (FileNotFoundError, AttributeError):
        raise ValueError("Test df file does not exist.")  # noqa: B904

    test_df = test_df[test_df["valid_ratio"] > valid_mask_threshold].reset_index(drop=True)  # type: ignore
    all_hex_ids = list(test_df["hex_id"].unique())
    for hex_id in all_hex_ids:
        print(f"======Working with hex{hex_id}========")
        one_hexel_df = test_df[test_df["hex_id"] == hex_id]
        hexel_indices = test_df[test_df["hex_id"] == hex_id].index.tolist()
        if len(str(hex_id)) != 2:
            hex_id = "0" + str(hex_id)
        hex_test_predictions = test_predictions[hexel_indices]
        reconstructed_hexel_denorm, gt_elevation_grid_profile = get_predicted_hexel(
            base_dir=data_dir,
            raw_data_dir=raw_data_dir,
            test_df=one_hexel_df,
            predictions=hex_test_predictions,
            min_target_val=min_target_val,
            max_target_val=max_target_val,
            hex_id=hex_id,
            modelling_approach=modelling_approach,
            out_norm=out_norm,
            stitch_mode="mean",
            win_h=128,
            win_w=128,
        )
        save_predicted_hexels(reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id, config.save_dir)
        # Save the hex as plt plot
        hex_dir = os.path.join(raw_data_dir, f"hex{hex_id}")
        fpath = find_simulation_output_file(hex_dir, hex_id, output_type, season=season, cause=cause)
        grid_gt = load_output_burn_grid(fpath)
        visualize_burn_prob_grid(gt_grid=grid_gt, pred_grid=reconstructed_hexel_denorm, hex_id=hex_id, save_dir=config.save_dir)
        print(f"=======Saved subplot for hex{hex_id}==============")

    print(f"=======Total Evaluation Time {round(time.time()-start_time, 3)}s========")
    print(f"=======Prediction Time {round(preds_time, 3)}s========")


if __name__ == "__main__":
    main()
