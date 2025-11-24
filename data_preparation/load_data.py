# TODO: delete later


import os

from data_preparation.ignition_grid import load_ignition_grid


def stack_grid():
    pass


if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    season = 1
    cause = 1

    out_ignition_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, "ignitions_module/ignition_grids"), season=season, cause=cause)
