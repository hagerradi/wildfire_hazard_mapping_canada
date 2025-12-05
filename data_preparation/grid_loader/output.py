import os

import numpy as np

from data_preparation.grid_loader.utils import (
    FIRE_INTENSITY_MAX,
    FIRE_INTENSITY_MIN,
    NODATA,
    load_raster,
    visualize_burn_prob_grid,
    visualize_fire_intensity_grid,
)


def load_output_fire_intensity_grid(path: str)-> np.ndarray:
    """Load elevation grid and normalize values."""
    output_fire_intensity_grid = load_raster(path)
    # normalize fire intensity data
    output_fire_intensity_grid = ((output_fire_intensity_grid - FIRE_INTENSITY_MIN) / (FIRE_INTENSITY_MAX - FIRE_INTENSITY_MIN))
    output_fire_intensity_grid = output_fire_intensity_grid.filled(NODATA)
    return output_fire_intensity_grid

def load_output_burn_prob_grid(path: str)-> np.ndarray:
    """Load burn probability grid."""
    output_burn_prob_grid = load_raster(path) #no norm because each pixel prob is independent of each other
    return output_burn_prob_grid.data

def load_output_burn_count_grid(path: str)-> np.ndarray:
    """Load burn count grid."""
    output_burn_count_grid = load_raster(path) # norm needed
    return output_burn_count_grid.data


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    out_fire_intensity_grid = load_output_fire_intensity_grid(path=os.path.join(root_dir, "outputs/hex_05_fiRaw_mean.tif"))  # noqa: F821
    print(out_fire_intensity_grid.shape)
    visualize_fire_intensity_grid(out_fire_intensity_grid)

    output_burn_prob_grid = load_output_burn_prob_grid(path=os.path.join(root_dir, "outputs/hex_05_20000_iter_bp.tif"))  # noqa: F821
    print(output_burn_prob_grid.shape)
    visualize_burn_prob_grid(output_burn_prob_grid)