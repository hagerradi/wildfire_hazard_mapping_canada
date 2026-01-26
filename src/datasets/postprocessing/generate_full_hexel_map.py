import argparse
import copy
import re
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LogNorm, Normalize
from rasterio.enums import Resampling

# The grid layout for the Canada map (Row, Col)
# Geometry: rows and columns are offset by 0.5 hexel
HEX_MAP_LAYOUT = {
    # Row 0 (top row of the grid)
    22: (0, 2),
    # Row 1
    20: (1, 1),
    23: (1, 3),
    # Row 2
    12: (2, 2),
    24: (2, 4),
    # Row 3
    21: (3, 1),
    10: (3, 3),
    # Row 4
    19: (4, 2),
    35: (4, 4),
    # Row 5
    25: (5, 1),
    13: (5, 3),
    34: (5, 5),
    # Row 6
    50: (6, 0),
    2: (6, 2),
    5: (6, 4),
    33: (6, 6),
    37: (6, 12),
    46: (6, 14),
    # Row 7
    26: (7, 1),
    3: (7, 3),
    1: (7, 5),
    36: (7, 7),
    43: (7, 11),
    15: (7, 13),
    # Row 8
    17: (8, 2),
    38: (8, 4),
    8: (8, 6),
    4: (8, 8),
    7: (8, 12),
    51: (8, 16),
    # Row 9
    27: (9, 1),
    39: (9, 3),
    29: (9, 5),
    18: (9, 7),
    14: (9, 9),
    9: (9, 11),
    45: (9, 13),
    # Row 10
    52: (10, 0),
    40: (10, 2),
    28: (10, 6),
    6: (10, 8),
    44: (10, 10),
    16: (10, 12),
    # Row 11
    41: (11, 7),
    31: (11, 9),
    11: (11, 11),
    48: (11, 13),
    53: (11, 15),
    # Row 12
    42: (12, 8),
    30: (12, 10),
    47: (12, 12),
    49: (12, 14),
    # Row 13 (most bottom row of the grid)
    32: (13, 11),
}


def get_hex_center(row: int, col: int, radius: float = 1.0) -> tuple[float, float]:
    """Maps (row, col) to center (x, y) coords. for a flat-topped hexagon."""
    x = col * (1.5 * radius)
    y = -row * (np.sqrt(3) / 2 * radius)
    return x, y


def get_flat_hex_vertices(center_x: float, center_y: float, radius: float) -> np.ndarray:
    """Gets array of coords. of the 6 corners for a flat-topped hexagon."""
    angles = np.radians([0, 60, 120, 180, 240, 300])
    x_offset = radius * np.cos(angles)
    y_offset = radius * np.sin(angles)
    return np.column_stack((center_x + x_offset, center_y + y_offset))


def create_geometric_mask(shape: tuple[int, int], vertices_normalized: np.ndarray) -> np.ndarray:
    """Creates a bool. mask for the raster based on normalized hexagon vertices."""
    h, w = shape
    y, x = np.mgrid[:h, :w]
    y = (y / h) * 2 - 1
    x = (x / w) * 2 - 1
    points = np.vstack((x.flatten(), -y.flatten())).T
    path = mpath.Path(vertices_normalized)
    mask = path.contains_points(points).reshape(h, w)
    return mask


def find_hex_files(folder: Path, pattern: str) -> dict[int, Path]:
    """Scans a folder for files matching the pattern."""
    files = list(folder.glob(pattern))
    file_map = {}
    for f in files:
        match = re.search(r"hex_?(\d+)", f.parent.parent.name)
        if match:
            file_map[int(match.group(1))] = f
    return file_map


def calculate_global_stats(file_map: dict[int, Path]) -> tuple[float, float]:
    """
    Returns (global_min_positive, global_max).
    """
    print(f"Scanning {len(file_map)} files for global statistics...")
    global_max = -np.inf
    global_min_pos = np.inf

    for f in file_map.values():
        try:
            with rasterio.open(f) as src:
                # Read low-res subsample for speed
                h_small = max(1, src.height // 10)
                w_small = max(1, src.width // 10)
                data = src.read(1, out_shape=(h_small, w_small), resampling=Resampling.max)

                if src.nodata is not None:
                    data = np.ma.masked_equal(data, src.nodata)

                if data.count() == 0:
                    continue

                cmax = data.max()
                if cmax > global_max:
                    global_max = cmax

                # For Log scale, find smallest positive non-zero
                valid_pos = data[data > 0]
                if valid_pos.size > 0:
                    cmin_pos = valid_pos.min()
                    if cmin_pos < global_min_pos:
                        global_min_pos = cmin_pos
        except Exception as e:
            print(f"Warning skipping stats for {f}: {e}")

    # fallbacks if inf. values
    if global_max == -np.inf:
        global_max = 1.0
    if global_min_pos == np.inf:
        global_min_pos = 1e-6

    return global_min_pos, global_max


def generate_stitched_map(
    data_folder_path: str,
    search_pattern: str,
    downsample_factor: int = 1,
    scale: str = "linear",
    show_hex_borders: bool = True,
    title: str = None,
    output_path: str = None,
):
    """
    Main function to generate the stiched full Canada map of hexels.

    Args:
        data_folder_path (str): Main dataset directory.
        search_pattern (str): File pattern to look for (default: bp maps).
        downsample_factor (int): Downsampling factor (for faster and lower-res. map).
        scale (str): Use 'log' for log norm. scale or 'linear' for linear min-max.
        show_hex_borders (bool): Whether to show hexel borders in the map.
        title (str): Figure title.
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

    # Get scale type selection for plotting
    if scale == "log":
        # For log scale, we use positive min. and global max.
        print(f"Using LOG scale: {pos_min:.2e} to {global_max:.2e}")
        norm = LogNorm(vmin=pos_min, vmax=global_max)
        final_title = title if title else "Canada Burn Probability (Log Scale)"
    elif scale == "linear":
        # For linear scale, we start at 0 prob/count.
        linear_min = 0
        print(f"Using LINEAR scale: {linear_min} to {global_max:.2e}")
        norm = Normalize(vmin=linear_min, vmax=global_max)  # type: ignore
        final_title = title if title else "Canada Burn Probability (Linear Scale)"
    else:
        raise ValueError(f"Unknown scale type: '{scale}'. Please use 'log' or 'linear'.")

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

                    # scale-specific visualization
                    if scale == "log":
                        # clamp zeros to min. positive
                        arr = np.maximum(arr, pos_min)
                    else:
                        # just mask negatives if any
                        arr = np.ma.masked_less(arr, 0)

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
    plt.title(final_title, fontsize=18)

    if im:
        cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
        cbar.set_label(f"Burn Probability ({scale.title()})", fontsize=12)

    if output_path:
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script to generate and save Canada burn probability map.")

    parser.add_argument(
        "--data-dir", type=str, default="/network/projects/amlrt/nrcan_wildfires/full_data/yan_bp3/", help="Path to base data directory."
    )

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

    parser.add_argument("--title", type=str, default="Canada Burn Probability Map", help="Custom plot title.")

    parser.add_argument("--output", type=str, default="experiments/full_map.png", help="Save to file instead of showing.")

    args = parser.parse_args()

    generate_stitched_map(
        data_folder_path=args.data_dir,
        search_pattern=args.pattern,
        downsample_factor=args.downsample,
        scale=args.scale,
        show_hex_borders=args.show_hex_borders,
        title=args.title,
        output_path=args.output,
    )
