import os

import numpy as np
import pandas as pd
import rasterio
from rasterio.profiles import Profile

from config import Config
from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.grid_loader.utils import denormalize_burn_count, denormalize_burn_prob, get_range_burn_count, get_range_burn_prob
from data_preparation.paths import ELEVATION_GRID_PATH
from data_preparation.utils import find_simulation_output_file
from datasets.postprocessing.visualize_predictions import visualize_burn_prob_grids
from logger import CometLogger
from src.datasets.postprocessing.stitch_hexel import stitch_windows


def save_predicted_hexels(predicted_hexel: np.ndarray, hexel_profile: Profile, hex_id: str, save_dir: str):
    """
    Save the predicted (reconstructed) hexel
    Args:
        predicted_hexel (np.ndarray) : 2d array of shape (height, width)
        hexel_profile (rasterio.profile): Profile for the hexel, required by rasterio for saving geospatial data
        hex_id (str): The id of the hex to be saved
        save_dir (str): directory to save the hexel
    """
    out_path = os.path.join(save_dir, "predicted_hexels", f"hexel_{hex_id}_predicted.tif")
    os.makedirs(os.path.join(save_dir, "predicted_hexels"), exist_ok=True)
    print("Shape of predicted array", predicted_hexel.shape)
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
    """
    Accumulate and stitch all the windows together to build the hexel
    """
    all_data_points, all_locations, all_masks = [], [], []
    for i, data in enumerate(np.array(df)):
        path = data[0]
        array = np.load(os.path.join(base_dir, path))[:, :, 0]
        mask = ~np.isnan(array)
        all_data_points.append(predictions[start_idx + i].reshape((win_h, win_w)))
        all_locations.append((data[5], data[6]))
        all_masks.append(mask.reshape((win_h, win_w)))
    reconstructed_hexel = stitch_windows(all_data_points, all_locations, all_masks, gt_shape, mode=stitch_mode)
    return reconstructed_hexel  # gt_shape


def get_predicted_hexel(
    base_dir: str,
    raw_data_dir: str,
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    min_target_val: float,
    max_target_val: float,
    hex_id: str,
    modelling_approach: str = "2",
    out_norm: str = "min_max",
    stitch_mode: str = "mean",
    win_h: int = 128,
    win_w: int = 128,
) -> tuple[np.ndarray, Profile]:
    """
    Returns the reconstructed hexel
    """
    start_idx = 0
    if os.path.exists(os.path.join(os.path.join(raw_data_dir, "hex" + str(hex_id)), ELEVATION_GRID_PATH)):
        with rasterio.open(os.path.join(os.path.join(raw_data_dir, "hex" + str(hex_id)), ELEVATION_GRID_PATH)) as src:
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
        reconstructed_hexel_denorm = denormalize_burn_prob(
            data=reconstructed_hexel, min_val=min_target_val, max_val=max_target_val, out_norm=out_norm
        )
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
            reconstructed_season_cause_hexel_denorm = denormalize_burn_count(
                data=reconstructed_season_cause_hexel, min_val=min_target_val, max_val=max_target_val
            )
            season_cause_hexels.append(reconstructed_season_cause_hexel_denorm)
            start_idx += len(filtered_season_cause_df)
        # merge the counts
        reconstructed_hexel_denorm = np.sum(np.stack(season_cause_hexels), axis=0)
        reconstructed_hexel_denorm = np.rint(reconstructed_hexel_denorm).astype("int32")
        gt_elevation_grid_profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

    return reconstructed_hexel_denorm, gt_elevation_grid_profile


def reconstruct_and_visualize_hexels(
    test_predictions: np.ndarray, config: Config, out_norm: str, experiment_logger: CometLogger | None = None
) -> None:
    """
    A util function to re-construct predicted hexels out of test predictions, and visualize side-by-side with the Groundtruth
    """
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

    # seperate hexels by their IDs
    test_df = test_df[test_df["valid_ratio"] > valid_mask_threshold].reset_index(drop=True)  # type: ignore
    all_hex_ids = list(test_df["hex_id"].unique())

    # loop over test hexels
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

        visualize_burn_prob_grids(
            gt_grid=grid_gt,
            pred_grid=reconstructed_hexel_denorm,
            hex_id=hex_id,
            save_dir=config.save_dir,
            experiment_logger=experiment_logger,
        )

        print(f"=======Saved subplot for hex{hex_id}==============")
