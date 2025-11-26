import os

import numpy as np

from data_preparation.utils import ELEV_NATIONAL_MAX, ELEV_NATIONAL_MIN, NODATA, load_raster
from data_preparation.visualize import visualize_elevation_grid


def load_elevation_grid(path: str)-> np.ndarray:
    """Load elevation grid and normalize values."""
    elevation_grid = load_raster(path)
    # normalize elevation grid data
    elevation_grid = (elevation_grid - ELEV_NATIONAL_MIN) / (ELEV_NATIONAL_MAX - ELEV_NATIONAL_MIN)
    elevation_grid = elevation_grid.filled(NODATA)
    return elevation_grid


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    out_elev_grid = load_elevation_grid(path=os.path.join(root_dir, "mapped_inputs/elev.asc"))  # noqa: F821
    visualize_elevation_grid(out_elev_grid)