import os

import numpy as np

from data_preparation.utils import NODATA, fuel_grouping, load_csv, load_raster
from data_preparation.visualize import visualize_fuel_grid


def group_fuels_in_grid(data: np.ndarray) -> np.ndarray:
    """Group classes of fuels based on fuel_grouping map"""
    group_names = list(fuel_grouping.keys())
    group_to_index = {name: i for i, name in enumerate(group_names)}

    class_to_group_index = {}
    for gname, codes in fuel_grouping.items():
        g_idx = group_to_index[gname]
        for cls in codes:
            class_to_group_index[cls] = g_idx

    # re-classify raster from FBP code to group index
    fuel_grid_data = np.full(data.shape, fill_value=NODATA, dtype=np.float32)

    for cls, g_idx in class_to_group_index.items():
        fuel_grid_data[data == cls] = g_idx

    # range of classes will be between 0 - len(fuel_grouping)
    return fuel_grid_data


def load_fuel_grid(path: str, fuel_table_path: str, group_fuels: bool = False)-> np.ndarray:
    """Load an FBP fuel raster and group fuel types if selected"""

    fuel_grid = load_raster(path)

    # option 1: group fuels
    if group_fuels:
        # convert classes to be in the range 0 - num_groups       
        return group_fuels_in_grid(data=fuel_grid)
    
    # option 2: keep grid as is, re-assign classes to be between 0 - num_classes
    fuel_table = load_csv(fuel_table_path)
    grid_values = fuel_table["grid_value"].tolist()
    class_to_index = {cls: i for i, cls in enumerate(grid_values)}

    fuel_grid_data = np.full(fuel_grid.shape, fill_value=NODATA, dtype=np.float32)
    for cls, idx in class_to_index.items():
        fuel_grid_data[fuel_grid == cls] = idx

    return fuel_grid_data


# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    out_fuel_grid = load_fuel_grid(path=os.path.join(root_dir, "mapped_inputs/fbp.asc"),
                                fuel_table_path=os.path.join(root_dir, "mapped_inputs/Fuel_table.lut"))  # noqa: F821
    
    print(np.unique_counts(out_fuel_grid))
    visualize_fuel_grid(out_fuel_grid)