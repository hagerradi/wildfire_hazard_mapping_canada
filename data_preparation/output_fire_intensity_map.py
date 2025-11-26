import os

import numpy as np

from data_preparation.utils import FIRE_INTENSITY_MAX, FIRE_INTENSITY_MIN, load_raster
from data_preparation.visualize import visualize_fire_intensity_grid


def load_output_fire_intensity_grid(path: str)-> np.ma.MaskedArray:
    """Load elevation grid and normalize values."""
    output_fire_intensity_grid = load_raster(path)
    # normalize elevation grid data
    output_fire_intensity_grid = ((output_fire_intensity_grid - FIRE_INTENSITY_MIN) / (FIRE_INTENSITY_MAX - FIRE_INTENSITY_MIN))
    return output_fire_intensity_grid


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    out_fire_intensity_grid = load_output_fire_intensity_grid(path=os.path.join(root_dir, "outputs/hex_05_fiRaw_mean.tif"))  # noqa: F821
    print(out_fire_intensity_grid.shape)
    visualize_fire_intensity_grid(out_fire_intensity_grid)