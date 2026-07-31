import argparse
import copy
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from rasterio.enums import Resampling

from datasets.postprocessing.full_map.utils import (
    HEX_MAP_LAYOUT,
    calculate_global_stats,
    create_geometric_mask,
    find_hex_files,
    get_flat_hex_vertices,
    get_hex_center,
    get_scale_settings,
)


def generate_stitched_map(
    data_folder_path: str,
    search_pattern: str,
    title: str,
    downsample_factor: int = 1,
    scale: str = "linear",
    show_hex_borders: bool = True,
    output_path: str = None,
    vmin: float = None,
    vmax: float = None,
):
    """
    Main function to generate the stiched full Canada map of hexels.

    Args:
        data_folder_path (str): Main dataset directory.
        search_pattern (str): File pattern to look for (default: bp maps).
        title (str): Figure title.
        downsample_factor (int): Downsampling factor (for faster and lower-res. map).
        scale (str): Use 'log' for log norm. scale or 'linear' for linear min-max.
        show_hex_borders (bool): Whether to show hexel borders in the map.
        output_path (str): Figure file output name for saving.
    """
    # Check data folder
    data_folder = Path(data_folder_path)
    if not data_folder.exists():
        print(f"Error: Data directory not found: {data_folder}")
        return

    # Find all hexel files to use for plotting based on pattern
    file_map = find_hex_files(data_folder, search_pattern)
    print(f"Found {len(file_map)} hex files.")
    if not file_map:
        return

    # get the min and max ranges for plotting
    pos_min, global_max = calculate_global_stats(file_map)

    # use min and max if provided
    if vmin is not None:
        pos_min = vmin
    if vmax is not None:
        global_max = vmax

    # Get scale type selection for plotting
    norm = get_scale_settings(scale, pos_min, global_max)

    # Setup plot
    _, ax = plt.subplots(figsize=(24, 18), dpi=300)
    target_radius = 1.0
    img_size = 2.0 * target_radius
    std_hex_verts = get_flat_hex_vertices(0, 0, 1.0)
    cached_mask = None

    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color="black", alpha=0)

    all_x, all_y = [], []
    im = None

    print(f"Stitching map ({scale.upper()} scale)...")

    # Iteration over all hexels and plotting
    for hex_id, (row, col) in HEX_MAP_LAYOUT.items():
        # get hexel center for plotting
        cx, cy = get_hex_center(row, col, radius=target_radius)
        all_x.append(cx)
        all_y.append(cy)

        # get polygon verticles for plotting
        if show_hex_borders:
            world_verts = get_flat_hex_vertices(cx, cy, target_radius)
            poly_patch = mpatches.Polygon(world_verts, closed=True, facecolor="none", edgecolor="red", linewidth=1.5, zorder=10, alpha=0.6)
            ax.add_patch(poly_patch)

        if hex_id in file_map:
            try:
                with rasterio.open(file_map[hex_id]) as src:
                    new_h = src.height // downsample_factor
                    new_w = src.width // downsample_factor
                    arr = src.read(1, out_shape=(new_h, new_w), resampling=Resampling.bilinear)

                    if src.nodata is not None:
                        arr = np.ma.masked_equal(arr, src.nodata)

                    # geometric mask
                    if cached_mask is None or cached_mask.shape != arr.shape:
                        cached_mask = create_geometric_mask(arr.shape, std_hex_verts)
                    arr = np.ma.masked_where(~cached_mask, arr)

                    arr = np.maximum(arr, pos_min) if scale == "log" else np.ma.masked_less(arr, 0)

                    extent = (cx - img_size / 2, cx + img_size / 2, cy - img_size / 2, cy + img_size / 2)
                    im = ax.imshow(arr, extent=extent, cmap=cmap, norm=norm, zorder=1)

            except Exception as e:
                print(f"Error hex {hex_id}: {e}")

        # overlay hexel number on top of the hexel
        ax.text(
            cx,
            cy,
            str(hex_id),
            ha="center",
            va="center",
            fontsize=9,
            color="white" if hex_id in file_map else "black",
            fontweight="bold",
            zorder=11,
        )

    if all_x:
        margin = 2.0
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)

    ax.set_aspect("equal")
    ax.axis("off")

    if title:
        plt.title(title, fontsize=18)

    if im:
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label(f"Burn Probability ({scale.title()})", fontsize=12)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Script to generate and save Canada burn probability map.")

    parser.add_argument("--data-dir", type=str, help="Path to base data directory.")

    parser.add_argument(
        "--pattern",
        type=str,
        default="hex*/outputs/*_iter_bp.tif",
        help="Glob pattern for TIF files to look for. For counts, use 'hex*/outputs/*_iter_bp.tif'.",
    )

    parser.add_argument(
        "--scale", type=str, choices=["log", "linear"], default="linear", help="Color scaling method: 'linear' (default) or 'log'."
    )

    parser.add_argument("--downsample", type=int, default=1, help="Downsample factor (int). Default 1.")

    parser.add_argument("--show_hex_borders", action="store_true", help="Show red hexel borders in the map.")

    parser.add_argument("--title", type=str, default=None, help="Custom plot title.")

    parser.add_argument("--output", type=str, default="experiments/full_map.png", help="Save to file instead of showing.")

    parser.add_argument("--vmin", type=float, default=None, help="Force minimum value for the color scale.")

    parser.add_argument("--vmax", type=float, default=None, help="Force maximum value for the color scale.")

    args = parser.parse_args()

    generate_stitched_map(
        data_folder_path=args.data_dir,
        search_pattern=args.pattern,
        title=args.title,
        downsample_factor=args.downsample,
        scale=args.scale,
        show_hex_borders=args.show_hex_borders,
        output_path=args.output,
        vmin=args.vmin,
        vmax=args.vmax,
    )


if __name__ == "__main__":
    main()
