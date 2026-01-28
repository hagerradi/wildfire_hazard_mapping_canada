import re
from pathlib import Path

import matplotlib.path as mpath
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
                data = src.read(1)

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


def get_scale_settings(scale: str, pos_min: float, global_max: float) -> Normalize:
    """Configures the normalization based on the scale type."""
    if scale == "log":
        # For log scale, we use positive min. and global max.
        print(f"Using LOG scale: {pos_min:.2e} to {global_max:.2e}")
        norm = LogNorm(vmin=pos_min, vmax=global_max)
    elif scale == "linear":
        # For linear scale, we start at 0 prob/count.
        linear_min = 0
        print(f"Using LINEAR scale: {linear_min} to {global_max:.2e}")
        norm = Normalize(vmin=linear_min, vmax=global_max)  # type: ignore
    else:
        raise ValueError(f"Unknown scale type: '{scale}'. Please use 'log' or 'linear'.")

    return norm
