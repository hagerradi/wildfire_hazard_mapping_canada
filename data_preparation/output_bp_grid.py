import os

import numpy as np
from utils import load_raster
from visualize import visualize_burn_prob_grid


def load_output_burn_prob_grid(path: str)-> np.ndarray:
    """Load elevation grid and normalize values."""
    output_burn_prob_grid = load_raster(path) #no norm because each pixel prob is independent of each other
    return output_burn_prob_grid.data


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    output_burn_prob_grid = load_output_burn_prob_grid(path=os.path.join(root_dir, "outputs/hex_05_20000_iter_bp.tif"))  # noqa: F821
    print(output_burn_prob_grid.shape)
    visualize_burn_prob_grid(output_burn_prob_grid)