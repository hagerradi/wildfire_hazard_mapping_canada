import os

import matplotlib.pyplot as plt
import numpy as np


def visualize_burn_prob_grid(gt_grid: np.ndarray, pred_grid: np.ndarray, hex_id: str, save_dir: str):
    """
    Visualizes Ground Truth and Predicted burn prob grids side-by-side.
    """

    out_path = os.path.join(save_dir, "predicted_hexels_plot", f"hexel_{hex_id}_predicted.png")
    os.makedirs(os.path.join(save_dir, "predicted_hexels_plot"), exist_ok=True)
    inferred_vmax = np.nanmax(gt_grid)
    # Create a figure with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Set the main title
    fig.suptitle(f"Burn Probability prediction Hex {hex_id}", fontsize=16)

    # Plot Ground Truth
    im1 = axes[0].imshow(gt_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[0].set_title("Ground Truth")
    axes[0].set_xlabel("Easting (m)")
    axes[0].set_ylabel("Northing (m)")

    # Plot Predicted
    im2 = axes[1].imshow(pred_grid, cmap="viridis", origin="upper", vmax=inferred_vmax)
    axes[1].set_title("Prediction")
    axes[1].set_xlabel("Easting (m)")
    # We can hide the ylabel for the second plot if they share the same axis
    axes[1].set_ylabel("Northing (m)")

    # Create a shared colorbar
    # cax puts the colorbar in its own space so it doesn't shrink the 2nd subplot
    cbar = fig.colorbar(im2, ax=axes.ravel().tolist(), label="Burn Probability", shrink=0.8)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")

    print(f"Figure saved to: {out_path}")
    plt.close(fig)  # Close the figure to free up memory (important for loops)
