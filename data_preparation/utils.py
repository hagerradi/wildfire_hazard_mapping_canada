import os
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

# value for nodata in the rasters
NODATA = -9999

# normalization values for elevation (on national scale)
ELEV_NATIONAL_MAX = 5855
ELEV_NATIONAL_MIN = -158

# Max wind velocity (TODO: need to modify when we have the entire dataset)
MAX_WIND_VELOCITY = 14.279999732971191

# Normalization values for Fire Intensity (TODO: need to rerun once we have the entire dataset)
FIRE_INTENSITY_MAX = 127247.0
FIRE_INTENSITY_MIN = 0.0

# mapping cause to cause index
fire_cause_mapping = {"h": 1, "l": 2}

# features of fire weather list to include
selected_weather_features = ['temp', 'rh', 'prec', 'ffmc', 'dmc','dc', 'isi', 'bui'] #'ws','wd_sin', 'wd_cos'

# grouping fuel classes
fuel_grouping = {
    "high":    [1, 2, 3, 4, 5, 6, 7, 650, 665],
    "medium":  [635],
    "low":     [11, 12, 13, 425, 525, 625],
    "grass":   [31, 32],
    "nonfuel": [101, 102, 106],
}

def load_raster(path: str) -> np.ma.MaskedArray:
    """ Load raster from given path"""
    if os.path.exists(path):  # noqa: F821
        with rasterio.open(path) as src:
            raster = src.read(1, masked=True) # mask out the nodata (-9999 values)
            return raster
    else:
        raise FileNotFoundError(f"File not found: {path}")
    

def load_csv(path: str) -> pd.DataFrame:
    """Load csv file from given path"""
    if os.path.exists(path):  # noqa: F821
        df = pd.read_csv(path)
        print("File loaded successfully.")
        return df
    else:
        raise FileNotFoundError(f"File not found: {path}")

def get_max_wind_velocity(data_path:str)->float:
    """Get the global maximum wind velocity for normalization"""
    all_hex = list(os.listdir(data_path))[1:]
    global_max_wind_velocity = -np.inf
    for hex in all_hex:
        path_wind_grids = f"./{data_path}/{hex}/burning_conditions_module/wind_grids"
        path_wind_grids = Path(path_wind_grids)
        all_wind_velocity_files = list(path_wind_grids.glob("w???_vel.asc"))
        for file_name in all_wind_velocity_files:
            wind_velocity_grid = load_raster(file_name)
            global_max_wind_velocity = max(global_max_wind_velocity, wind_velocity_grid.data.max())
    return (float(global_max_wind_velocity))

def get_range_output_fire_intensity(data_path:str)->tuple[float, float]:
    """Get the maximum and minimum output fire intensity for normalization"""
    all_hex = list(os.listdir(data_path))[1:]
    min_fire_intensity, max_fire_intensity = np.inf, -np.inf
    for hex in all_hex:
        path_output_files = f"{data_path}/{hex}/outputs/hex_{hex[3:]}_fiRaw_mean.tif"
        output_fire_intensity_grid = load_raster(path_output_files)
        max_fire_intensity = max(max_fire_intensity, output_fire_intensity_grid.max())
        min_fire_intensity = min(min_fire_intensity, output_fire_intensity_grid.min())
    return float(max_fire_intensity), float(min_fire_intensity)