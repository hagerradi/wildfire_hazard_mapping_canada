"""
End-to-end script for evaluation of one hexel
"""

import argparse
import os

import numpy as np
import pandas as pd
import rasterio
import yaml

from data_preparation.grid_loader import load_elevation_grid
from data_preparation.grid_loader.utils import denormalize_burn_count, get_range_burn_count, get_range_burn_prob
from data_preparation.paths import ELEVATION_GRID_PATH
from src.config import Config
from src.datasets.dataloader import get_test_loader
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.trainer import Trainer


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


def save_predicted_hexels(predicted_hexel, hexel_profile, hex_id, save_dir):
    out_path = os.path.join(save_dir, "predicted_hexels", f"hexel_{hex_id}_predicted.tif")
    os.makedirs(os.path.join(save_dir, "predicted_hexels"), exist_ok=True)
    print("Shape of ^redicted array", predicted_hexel.shape)
    with rasterio.open(out_path, "w", **hexel_profile) as dst:
        dst.write(predicted_hexel, 1)


def get_stitched_windows(
    base_dir: str,
    df: pd.DataFrame,
    predictions: np.ndarray,
    start_idx: int,
    gt_shape: tuple,
    stitch_mode: str = "mean",
    win_h: int = 128,
    win_w: int = 128,
) -> np.ndarray:
    all_data_points, all_locations, all_masks = [], [], []
    print(start_idx, len(np.array(df)), len(predictions))
    for i, data in enumerate(np.array(df)):
        path = data[0]
        array = np.load(os.path.join(base_dir, path))[:, :, 0]
        mask = ~np.isnan(array)
        all_data_points.append(predictions[start_idx + i].reshape((win_h, win_w)))
        all_locations.append((data[5], data[6]))
        all_masks.append(mask.reshape((win_h, win_w)))
    reconstructed_hexel = stitch_windows(all_data_points, all_locations, all_masks, gt_shape, mode=stitch_mode)
    return reconstructed_hexel


def get_predicted_hexel(
    base_dir: str,
    root_dir: str,
    predictions: np.ndarray,
    min_burn_val: float,
    max_burn_val: float,
    modelling_approach: str = "2",
    stitch_mode: str = "mean",
    win_h: int = 128,
    win_w: int = 128,
) -> tuple[np.ndarray, rasterio.profile, str]:
    test_df = pd.read_csv(os.path.join(base_dir, "test_indices.csv"))
    test_df = test_df[test_df["valid_ratio"] != 0.0]  # type: ignore
    hex_id = str(test_df["hex_id"].iloc[0])
    start_idx = 0
    if os.path.exists(os.path.join(os.path.join(root_dir, "hex" + str(hex_id)), ELEVATION_GRID_PATH)):
        with rasterio.open(os.path.join(os.path.join(root_dir, "hex" + str(hex_id)), ELEVATION_GRID_PATH)) as src:
            gt_elevation_grid = src.read(1, masked=True)
            gt_elevation_grid_profile = src.profile.copy()

    if modelling_approach == "1":
        reconstructed_hexel = get_stitched_windows(
            base_dir=base_dir,
            df=test_df,
            predictions=predictions,
            start_idx=start_idx,
            gt_shape=tuple(gt_elevation_grid.data.shape),
            stitch_mode=stitch_mode,
            win_h=win_h,
            win_w=win_w,
        )
        reconstructed_hexel_denorm = denormalize_burn_count(data=reconstructed_hexel, min_val=min_burn_val, max_val=max_burn_val)
        gt_elevation_grid_profile.update(dtype="float32", compress="lzw", nodata=-9999)  # type: ignore
    else:
        unique_season_cause = list(set(zip(test_df["season"], test_df["cause"])))
        season_cause_hexels = []
        for season, cause in unique_season_cause:
            filtered_season_cause_df = test_df[(test_df["season"] == season) & (test_df["cause"] == cause)]
            reconstructed_season_cause_hexel = get_stitched_windows(
                base_dir=base_dir,
                df=filtered_season_cause_df,
                predictions=predictions,
                start_idx=start_idx,
                gt_shape=tuple(gt_elevation_grid.data.shape),
                stitch_mode=stitch_mode,
                win_h=win_h,
                win_w=win_w,
            )
            print(f"{season}_{cause} reconstruction {np.unique(reconstructed_season_cause_hexel)}")
            reconstructed_season_cause_hexel_denorm = denormalize_burn_count(
                data=reconstructed_season_cause_hexel, min_val=min_burn_val, max_val=max_burn_val
            )
            print(f"{season}_{cause} reconstruction {np.unique(reconstructed_season_cause_hexel_denorm)}")
            season_cause_hexels.append(reconstructed_season_cause_hexel_denorm)
            start_idx += len(filtered_season_cause_df)
        # merge the counts
        reconstructed_hexel_denorm = np.sum(np.stack(season_cause_hexels), axis=0)
        reconstructed_hexel_denorm = np.rint(reconstructed_hexel_denorm).astype("int32")
        gt_elevation_grid_profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

    # Save the hexels
    print(np.unique(reconstructed_hexel_denorm))
    return reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    config.logger.enabled = False

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

    print("\n[Test metrics]")
    if isinstance(test_metrics, dict):
        for k, v in test_metrics.items():
            print(f"  {k}: {v:.6f}")

    data_dir = config.data.root_dir
    root_dir = "../yan_bp3"
    modelling_approach = "1"
    if modelling_approach == "1":
        max_burn_val, min_burn_val = get_range_burn_prob(root_dir="../yan_bp3")
    else:
        max_burn_val, min_burn_val = get_range_burn_count(root_dir="../yan_bp3")

    print("Min max val", max_burn_val, min_burn_val)
    if isinstance(test_predictions, str):
        # Handle the error or raise an exception
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")
    reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id = get_predicted_hexel(
        data_dir, root_dir, test_predictions, min_burn_val, max_burn_val, modelling_approach, stitch_mode="mean", win_h=128, win_w=128
    )
    save_predicted_hexels(reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id, config.save_dir)
    # TODO: Stitch predictions back to hexel
    # TODO: Merge predictions of multiple scenarios for approach 2
    # TODO: re-compute metrics at the hexel level


if __name__ == "__main__":
    main()
