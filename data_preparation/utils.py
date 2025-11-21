import os

import numpy as np
import pandas as pd
import rasterio

# value for nodata in the rasters
NODATA = -9999

# normalization values for elevation (on national scale)
ELEV_NATIONAL_MAX = 5855
ELEV_NATIONAL_MIN = -158

# mapping cause to cause index
fire_cause_mapping = {1: "h", 2: "l"}

# features of fire weather list to include
selected_weather_features = ['temp', 'rh', 'ws','wd_sin', 'wd_cos', 'prec', 'ffmc', 'dmc','dc', 'isi', 'bui']

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
