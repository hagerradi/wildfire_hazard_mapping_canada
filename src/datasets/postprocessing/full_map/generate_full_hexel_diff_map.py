import argparse
import copy
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.enums import Resampling

from datasets.postprocessing.full_map.utils import (
    HEX_MAP_LAYOUT,
    create_geometric_mask,
    find_hex_files,
    get_flat_hex_vertices,
    get_hex_center,
)


def generate_diff_map(
    target_dir: str,
    target_pattern: str,
    pred_dir: str,
    pred_pattern: str,
    title: str,
    downsample_factor: int = 1,
    show_hex_borders: bool = True,
    output_path: str = None,
):
    target_folder = Path(target_dir)
    pred_folder = Path(pred_dir)

    # find all preds and targets hexels
    target_map = find_hex_files(target_folder, target_pattern)
    pred_map = find_hex_files(pred_folder, pred_pattern)

    # only process hexels that are in both sets
    common_hexes = set(target_map.keys()).intersection(set(pred_map.keys()))
    print(f"Found {len(common_hexes)} overlapping hexels for diff mapping.")
    if not common_hexes:
        print("Error: No matching hexels found between targets and predictions.")
        return

    _, ax = plt.subplots(figsize=(24, 18), dpi=300)
    target_radius = 1.0
    img_size = 2.0 * target_radius
    std_hex_verts = get_flat_hex_vertices(0, 0, 1.0)
    cached_mask = None

    cmap = copy.copy(plt.get_cmap("RdBu_r"))
    cmap.set_bad(color="black", alpha=0)

    all_x, all_y = [], []
    im = None

    global_max_diff = 0.0

    print("Calculating diffs and stitching map...")

    # get max error for symmetrical color scaling
    for hex_id in common_hexes:
        with rasterio.open(target_map[hex_id]) as src_tgt, rasterio.open(pred_map[hex_id]) as src_prd:
            arr_tgt = src_tgt.read(1)
            arr_prd = src_prd.read(1)

            if src_tgt.nodata is not None:
                arr_tgt = np.ma.masked_equal(arr_tgt, src_tgt.nodata)
            if src_prd.nodata is not None:
                arr_prd = np.ma.masked_equal(arr_prd, src_prd.nodata)

            valid_mask = ~np.ma.getmaskarray(arr_tgt) & ~np.ma.getmaskarray(arr_prd)
            if np.any(valid_mask):
                diff = arr_prd[valid_mask] - arr_tgt[valid_mask]
                global_max_diff = max(global_max_diff, np.max(np.abs(diff)))

    # create a symmetrical norm centered at exactly 0.0
    max_val = max(global_max_diff, 1e-6)
    norm = TwoSlopeNorm(vmin=-max_val, vcenter=0.0, vmax=max_val)

    # plotting
    for hex_id, (row, col) in HEX_MAP_LAYOUT.items():
        cx, cy = get_hex_center(row, col, radius=target_radius)
        all_x.append(cx)
        all_y.append(cy)

        # draw borders
        if show_hex_borders:
            world_verts = get_flat_hex_vertices(cx, cy, target_radius)
            poly_patch = mpatches.Polygon(
                world_verts, closed=True, facecolor="none", edgecolor="black", linewidth=1.0, zorder=10, alpha=0.3
            )
            ax.add_patch(poly_patch)

        if hex_id in common_hexes:
            try:
                with rasterio.open(target_map[hex_id]) as src_tgt, rasterio.open(pred_map[hex_id]) as src_prd:
                    new_h = src_tgt.height // downsample_factor
                    new_w = src_tgt.width // downsample_factor

                    arr_tgt = src_tgt.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear)
                    arr_prd = src_prd.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear)

                    # masking of nodata
                    if src_tgt.nodata is not None:
                        arr_tgt = np.ma.masked_equal(arr_tgt, src_tgt.nodata)
                    if src_prd.nodata is not None:
                        arr_prd = np.ma.masked_equal(arr_prd, src_prd.nodata)

                    # preds - targets
                    diff_arr = arr_prd - arr_tgt

                    # geometric mask
                    if cached_mask is None or cached_mask.shape != diff_arr.shape:
                        cached_mask = create_geometric_mask(diff_arr.shape, std_hex_verts)
                    diff_arr = np.ma.masked_where(~cached_mask, diff_arr)

                    extent = (cx - img_size / 2, cx + img_size / 2, cy - img_size / 2, cy + img_size / 2)
                    im = ax.imshow(diff_arr, extent=extent, cmap=cmap, norm=norm, zorder=1)

            except Exception as e:
                print(f"Error hex {hex_id}: {e}")

        # hexel numbering
        ax.text(cx, cy, str(hex_id), ha="center", va="center", fontsize=9, color="black", fontweight="bold", zorder=11, alpha=0.5)

    if all_x:
        margin = 2.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal")
    ax.axis("off")
    plt.title(title, fontsize=18)

    if im:
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label("Burn Probability Error (Prediction - Target)", fontsize=12)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight", facecolor="white")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Generate a Difference Map (Pred - Target)")
    parser.add_argument("--target-dir", type=str, required=True, help="Path to ground truth targets.")
    parser.add_argument("--target-pattern", type=str, default="hex*/outputs/*_iter_bp.tif")
    parser.add_argument("--pred-dir", type=str, required=True, help="Path to model predictions.")
    parser.add_argument("--pred-pattern", type=str, default="*predicted.tif")
    parser.add_argument("--downsample", type=int, default=1)
    parser.add_argument("--title", type=str, default=None)
    parser.add_argument("--output", type=str, default="experiments/diff_map.png")
    args = parser.parse_args()

    generate_diff_map(
        target_dir=args.target_dir,
        target_pattern=args.target_pattern,
        pred_dir=args.pred_dir,
        pred_pattern=args.pred_pattern,
        title=args.title,
        downsample_factor=args.downsample,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
