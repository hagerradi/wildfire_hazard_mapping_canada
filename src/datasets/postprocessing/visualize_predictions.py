import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

from config import Config
from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.grid_loader.utils import get_range_burn_count, get_range_burn_prob
from data_preparation.utils import find_simulation_output_file
from datasets.postprocessing.utils import get_predicted_hexel, save_predicted_hexels
from logger import CometLogger


def visualize_predicted_hexels(
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


def visualize_burn_prob_grids(
    gt_grid: np.ndarray, pred_grid: np.ndarray, hex_id: str, save_dir: str, experiment_logger: CometLogger = None
):
    """
    Visualizes Ground Truth, Prediction, and Difference (GT - Prediction), side-by-side.
    """

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hexel_{hex_id}_predicted.png")

    valid_mask = np.isfinite(gt_grid)
    diff_grid = np.where(valid_mask, gt_grid - pred_grid, np.nan)
    # Shared scale for GT and Prediction
    shared_vmin = np.nanmin([np.nanmin(gt_grid), np.nanmin(pred_grid)])
    shared_vmax = np.nanmax([np.nanmax(gt_grid), np.nanmax(pred_grid)])

    # Symmetric scale for difference around 0
    diff_abs_max = np.nanmax(np.abs(diff_grid))
    diff_norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0.0, vmax=diff_abs_max)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    fig.suptitle(f"Burn Probability Prediction — Hex {hex_id}", fontsize=16)

    # --- Ground Truth ---
    im1 = axes[0].imshow(
        gt_grid,
        cmap="viridis",
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # --- Prediction ---
    im2 = axes[1].imshow(
        pred_grid,
        cmap="viridis",
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")

    # --- Difference ---
    im3 = axes[2].imshow(
        diff_grid,
        cmap="RdBu_r",
        origin="upper",
        norm=diff_norm,
    )
    axes[2].set_title("Difference (GT - Prediction)")
    axes[2].set_xlabel("Easting (m)")
    axes[2].set_ylabel("Northing (m)")

    # Shared colorbar for first two plots only
    cbar_shared = fig.colorbar(
        im2,
        ax=axes[:2],
        shrink=0.85,
        pad=0.02,
    )
    cbar_shared.set_label("Burn Probability")

    # Separate colorbar for difference plot only
    cbar_diff = fig.colorbar(
        im3,
        ax=axes[2],
        shrink=0.85,
        pad=0.02,
    )
    cbar_diff.set_label("Difference")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Figure saved to: {out_path}")
    if experiment_logger:
        experiment_logger.log_image(
            image_path=out_path,
            name=f"predicted_hexel_{hex_id}",
        )

    plt.close(fig)
