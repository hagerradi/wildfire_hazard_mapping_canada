import functools
import os
from typing import Callable

import numpy as np
import pandas as pd
import rasterio
import torch
from rasterio.profiles import Profile

from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.grid_loader.utils import denormalize_burn_count, denormalize_burn_prob, get_range_burn_count, get_range_burn_prob
from data_preparation.paths import ELEVATION_GRID_PATH
from data_preparation.utils import find_simulation_output_file
from src.config import Config
from src.datasets.postprocessing.stitch_hexel import stitch_windows
from src.datasets.postprocessing.visualize_predictions import visualize_burn_prob_grids, visualize_hexel_iou
from src.logger import CometLogger
from src.trainer import Trainer
from src.utils import AVAILABLE_METRICS


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
    modelling_approach: str = "1",
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

    # clip predictions between 0 and 1 in case of outliers
    predictions = np.clip(predictions, 0, 1)

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
        unique_season_cause = list(set(zip(test_df["season"], test_df["cause"], strict=False)))
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
        # clip values to the true range, in case of outliers
        reconstructed_hexel_denorm = np.clip(reconstructed_hexel_denorm, min_target_val, max_target_val)
        gt_elevation_grid_profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

    return reconstructed_hexel_denorm, gt_elevation_grid_profile


def calculate_hexel_metrics_pytorch(
    gt_grid: np.ndarray, pred_grid: np.ndarray, device: torch.device, metric_functions: dict[str, Callable], noise_threshold: float = 1e-4
) -> dict[str, float]:
    """
    Utils to convert 2D numpy hexels into torch tensors to run the global per-hexel eval. metrics.
    """
    valid_mask_np = ~np.isnan(gt_grid) & ~np.isnan(pred_grid)
    valid_mask_np = valid_mask_np & (gt_grid >= 0.0)

    gt_clean = np.nan_to_num(gt_grid, nan=0.0)
    pred_clean = np.nan_to_num(pred_grid, nan=0.0)

    # clamp background noise to avoid spearman ranking issue
    gt_clean[gt_clean < noise_threshold] = 0.0
    pred_clean[pred_clean < noise_threshold] = 0.0

    t_targets = torch.from_numpy(gt_clean).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_preds = torch.from_numpy(pred_clean).to(device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    t_mask = torch.from_numpy(valid_mask_np).to(device=device, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    results = {}
    # compute metrics requested in config.
    with torch.no_grad():
        for name, metric_fn in metric_functions.items():
            val = metric_fn(t_preds, t_targets, t_mask)
            results[name] = val.item()

    return results


def get_hexel_binary_maps(pred_grid: np.ndarray, gt_grid: np.ndarray, percentile: float = 0.95):
    """
    Utils to get the Top K percentile thresholds (binary maps) for full 2D numpy hexel grids.
    """
    valid_mask = ~np.isnan(gt_grid) & ~np.isnan(pred_grid)

    p_valid = pred_grid[valid_mask]
    t_valid = gt_grid[valid_mask]

    pred_bin = np.zeros_like(pred_grid, dtype=bool)
    gt_bin = np.zeros_like(gt_grid, dtype=bool)

    n_valid = len(p_valid)
    if n_valid > 0:
        # get count (number of elements) for the specific top K %
        k = int(np.ceil(percentile * n_valid))
        if k >= n_valid:
            pred_bin[valid_mask] = True
            gt_bin[valid_mask] = True
        elif k > 0:
            # select exactly k highest values within the valid area
            pred_valid_bin = np.zeros_like(p_valid, dtype=bool)
            gt_valid_bin = np.zeros_like(t_valid, dtype=bool)
            pred_topk_idx = np.argpartition(p_valid, -k)[-k:]
            gt_topk_idx = np.argpartition(t_valid, -k)[-k:]
            pred_valid_bin[pred_topk_idx] = True
            gt_valid_bin[gt_topk_idx] = True
            pred_bin[valid_mask] = pred_valid_bin
            gt_bin[valid_mask] = gt_valid_bin


def evaluate_and_visualize_hexels(
    test_predictions: np.ndarray,
    config: Config,
    out_norm: str,
    experiment_logger: CometLogger | None = None,
    trainer: Trainer | None = None,
) -> dict[str, float]:
    """
    A util function to re-construct predicted hexels out of test predictions, and visualize side-by-side with the Groundtruth.
    Also computes global stitched hexel-level metrics if a trainer is provided.
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
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    try:
        test_df = pd.read_csv(os.path.join(data_dir, config.data.test_split))
    except (FileNotFoundError, AttributeError):
        raise ValueError("Test df file does not exist.")  # noqa: B904

    # separate hexels by their IDs
    test_df = test_df[test_df["valid_ratio"] > valid_mask_threshold].reset_index(drop=True)  # type: ignore
    all_hex_ids = list(test_df["hex_id"].unique())

    # init. hexel metrics
    all_hexel_metrics = []

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

        # when we provide trainer, it will trigger global hexel-level metrics
        if trainer is not None:
            hex_metrics = calculate_hexel_metrics_pytorch(grid_gt, reconstructed_hexel_denorm, trainer.device, trainer.metric_functions)
            all_hexel_metrics.append(hex_metrics)

            # get top k perc. values dynamically
            percentiles_to_plot = [
                fn.keywords["percentile"]
                for _, fn in trainer.metric_functions.items()
                if isinstance(fn, functools.partial) and "percentile" in fn.keywords
            ]

            # generate the TopK IoU plots
            for p in percentiles_to_plot:
                pred_bin, gt_bin = get_hexel_binary_maps(reconstructed_hexel_denorm, grid_gt, percentile=p)
                visualize_hexel_iou(grid_gt, reconstructed_hexel_denorm, gt_bin, pred_bin, hex_id, config.save_dir, p)

        print(f"=======Saved subplots for hex{hex_id}==============")

    # aggregate final scores
    final_global_metrics = {}
    if trainer is not None and len(all_hexel_metrics) > 0:
        # average across all hexels
        for key in trainer.metric_functions.keys():
            mean_val = np.nanmean([hm[key] for hm in all_hexel_metrics if key in hm and not np.isnan(hm[key])])
            final_global_metrics[key] = float(mean_val)

    return final_global_metrics
