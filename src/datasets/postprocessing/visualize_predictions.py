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

    inferred_vmax = max(np.nanmax(gt_grid), np.nanmax(pred_grid))

    # plot the full targets and preds maps
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f"Top {top_pct_str}% Burn Probability Hotspots - Hex {hex_id}", fontsize=16)

    _ = axes[0, 0].imshow(pred_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[0, 0].set_title("Prediction")
    axes[0, 0].set_xlabel("Easting (m)")
    axes[0, 0].set_ylabel("Northing (m)")

    im2 = axes[0, 1].imshow(gt_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[0, 1].set_title("Ground Truth")
    axes[0, 1].set_xlabel("Easting (m)")
    axes[0, 1].set_ylabel("Northing (m)")

    fig.colorbar(im2, ax=axes.ravel().tolist(), label="Burn Probability", shrink=0.6)

    # get the thresholded binary preds and targets maps
    gt_bin_viz = np.where(np.isnan(gt_grid), np.nan, gt_bin.astype(float))
    pred_bin_viz = np.where(np.isnan(pred_grid), np.nan, pred_bin.astype(float))

    # plot the binary top K preds and targets maps
    _ = axes[1, 0].imshow(pred_bin_viz, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1, 0].set_title(f"Prediction (Top {top_pct_str}%)")
    axes[1, 0].set_xlabel("Easting (m)")
    axes[1, 0].set_ylabel("Northing (m)")

    _ = axes[1, 1].imshow(gt_bin_viz, cmap="Reds", origin="upper", vmin=0, vmax=1)
    axes[1, 1].set_title(f"Ground Truth (Top {top_pct_str}%)")
    axes[1, 1].set_xlabel("Easting (m)")
    axes[1, 1].set_ylabel("Northing (m)")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_hexbin_distribution(
    gt_grid: np.ndarray, pred_grid: np.ndarray, hex_id: str | int, save_dir: str, experiment_logger: CometLogger | None = None
) -> None:
    """
    Generates and saves a 2D hex binning histogram comparing preds vs. target probabilities.
    """
    hex_id_str = str(hex_id).zfill(2)

    valid_mask = ~np.isnan(gt_grid) & ~np.isnan(pred_grid)
    gt_vals = gt_grid[valid_mask]
    pred_vals = pred_grid[valid_mask]

    # set dynamic max limit and default fallback
    if len(gt_vals) > 0 and len(pred_vals) > 0:
        actual_max = float(max(np.max(gt_vals), np.max(pred_vals)))
        max_limit = min(1.0, actual_max * 1.05)
        if max_limit == 0.0:
            max_limit = 0.15
    else:
        max_limit = 0.15

    fig, ax = plt.subplots(figsize=(8, 7))
    hb = ax.hexbin(gt_vals, pred_vals, gridsize=100, cmap="viridis", bins="log", mincnt=1)

    ax.plot([0, max_limit], [0, max_limit], color="black", linestyle="--", linewidth=2, label="Perfect Alignment")

    ax.set_title(f"Probabilities Distribution: Preds vs Targets (GT) - Hex {hex_id_str}")
    ax.set_xlabel("Ground Truth Probability")
    ax.set_ylabel("Predicted Probability")

    ax.set_xlim(0, max_limit)
    ax.set_ylim(0, max_limit)

    fig.colorbar(hb, ax=ax, label="Log(Count of Pixels)")
    ax.legend()

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_hexbin_path = os.path.join(out_dir, f"hexbin_hex_{hex_id_str}.png")

    plt.savefig(out_hexbin_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    if experiment_logger is not None:
        experiment_logger.log_image(out_hexbin_path, name=f"hexbin_hex_{hex_id_str}")


def plot_histogram_distribution(
    gt_grid: np.ndarray,
    pred_grid: np.ndarray,
    hex_id: str | int,
    save_dir: str,
    experiment_logger: CometLogger | None = None,
    num_bins: int = 100,
) -> None:
    """
    Generates and saves an overlaid 1D histogram comparing the global distributions
    of preds and targets probs on a log. scale.
    """
    hex_id_str = str(hex_id).zfill(2)

    valid_mask = ~np.isnan(gt_grid) & ~np.isnan(pred_grid)
    gt_vals = gt_grid[valid_mask]
    pred_vals = pred_grid[valid_mask]

    # set dynamic max limit and default fallback
    if len(gt_vals) > 0 and len(pred_vals) > 0:
        max_val = float(max(np.max(gt_vals), np.max(pred_vals)))
        if max_val == 0.0:
            max_val = 0.15
    else:
        max_val = 0.15

    shared_bins = np.linspace(0.0, max_val, num=num_bins)
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

    ax.hist(gt_vals, bins=shared_bins.tolist(), color="blue", alpha=0.5, log=True, label="Ground Truth")
    ax.hist(pred_vals, bins=shared_bins.tolist(), color="orange", alpha=0.5, log=True, label="Prediction")

    ax.set_title(f"Overlayed Input Distributions (Log Scale) - Hex {hex_id_str}", fontsize=14)
    ax.set_xlabel("Burn Probability", fontsize=12)
    ax.set_ylabel("Pixel Count (Log Scale)", fontsize=12)

    ax.legend(fontsize=12)

    out_dir = os.path.join(save_dir, "predicted_hexels_plot")
    os.makedirs(out_dir, exist_ok=True)
    out_hist_path = os.path.join(out_dir, f"hist_hex_{hex_id_str}.png")

    plt.savefig(out_hist_path, bbox_inches="tight")
    plt.close(fig)

    if experiment_logger is not None:
        experiment_logger.log_image(out_hist_path, name=f"hist_dist_hex_{hex_id_str}")
