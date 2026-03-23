import os

import numpy as np

from data_preparation.grid_loader.utils import NODATA, load_raster, visualize_elevation_grid


def load_elevation_grid(path: str) -> np.ndarray:
    """Load elevation grid and normalize values."""
    masked_grid = load_raster(path)
    float_masked_grid = masked_grid.astype(np.float32)
    elevation_grid = np.ma.filled(float_masked_grid, fill_value=NODATA)
    return elevation_grid


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    out_elev_grid = load_elevation_grid(path=os.path.join(root_dir, "mapped_inputs/elev.asc"))  # noqa: F821
    visualize_elevation_grid(out_elev_grid)
