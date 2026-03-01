import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from src.logger import CometLogger


def visualize_burn_prob_grids(
    gt_grid: np.ndarray, pred_grid: np.ndarray, hex_id: str, save_dir: str, experiment_logger: CometLogger | None = None
):
    """
    Visualizes Ground Truth, Prediction, and Difference (GT - Prediction), side-by-side.
    """

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hexel_{hex_id}_predicted.png")

    valid_mask = np.isfinite(gt_grid) & np.isfinite(pred_grid)
    gt_grid = np.where(valid_mask, gt_grid, np.nan)
    pred_grid = np.where(valid_mask, pred_grid, np.nan)
    diff_grid = np.where(valid_mask, pred_grid - gt_grid, np.nan)
    # Shared scale for GT and Prediction
    shared_vmin = np.nanmin([np.nanmin(gt_grid), np.nanmin(pred_grid)])
    shared_vmax = np.nanmax([np.nanmax(gt_grid), np.nanmax(pred_grid)])

    # Symmetric scale for difference around 0
    diff_abs_max = np.nanmax(np.abs(diff_grid))
    diff_norm = TwoSlopeNorm(vmin=-diff_abs_max, vcenter=0.0, vmax=diff_abs_max)

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), constrained_layout=True)
    fig.suptitle(f"Burn Probability Prediction — Hex {hex_id}", fontsize=16)

    # --- Prediction ---
    im2 = axes[0].imshow(
        pred_grid,
        cmap="viridis",
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[0].set_title("Prediction")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # --- Ground Truth ---
    im1 = axes[1].imshow(
        gt_grid,
        cmap="viridis",
        origin="upper",
        vmin=shared_vmin,
        vmax=shared_vmax,
    )
    axes[1].set_title("Ground Truth")
    axes[1].set_xlabel("Easting (m)")
    axes[1].set_ylabel("Northing (m)")

    # --- Difference ---
    im3 = axes[2].imshow(
        diff_grid,
        cmap="RdBu_r",
        origin="upper",
        norm=diff_norm,
    )
    axes[2].set_title("Difference (Prediction - GT)")
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


def visualize_hexel_iou(
    gt_grid: np.ndarray, pred_grid: np.ndarray, gt_bin: np.ndarray, pred_bin: np.ndarray, hex_id: str, save_dir: str, percentile: float
):
    """
    Visualizes targets and preds burn prob. maps alongside their binary Top-K hotspots.
    """
    top_pct = round((1.0 - percentile) * 100.0, 2)
    top_pct_str = f"{top_pct:g}"

    # we save them in same dir. as the predicted hexel plots
    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"hexel_{hex_id}_top_{top_pct_str}perc_iou.png")

    inferred_vmax = np.nanmax(gt_grid)

    # plot the full targets and preds maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Top {top_pct_str}% Burn Probability Hotspots - Hex {hex_id}", fontsize=16)

    _ = axes[0, 0].imshow(gt_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[0, 0].set_title("Ground Truth")
    axes[0, 0].set_xlabel("Easting (m)")
    axes[0, 0].set_ylabel("Northing (m)")

    im2 = axes[0, 1].imshow(pred_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[0, 1].set_title("Prediction")
    axes[0, 1].set_xlabel("Easting (m)")
    axes[0, 1].set_ylabel("Northing (m)")

    fig.colorbar(im2, ax=axes[0, :].ravel().tolist(), label="Burn Probability", shrink=0.8)

    # get the thresholded binary preds and targets maps
    gt_bin_viz = np.where(np.isnan(gt_grid), np.nan, gt_bin.astype(float))
    pred_bin_viz = np.where(np.isnan(pred_grid), np.nan, pred_bin.astype(float))

    # plot the binary top K preds and targets maps
    _ = axes[1, 0].imshow(gt_bin_viz, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Ground Truth (Top {top_pct_str}%)")
    axes[1, 0].set_xlabel("Easting (m)")
    axes[1, 0].set_ylabel("Northing (m)")

    _ = axes[1, 1].imshow(pred_bin_viz, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Prediction (Top {top_pct_str}%)")
    axes[1, 1].set_xlabel("Easting (m)")
    axes[1, 1].set_ylabel("Northing (m)")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
