import numpy as np
import rasterio

# value for nodata in the rasters
NODATA = -9999

# mapping cause to cause index
fire_cause_mapping = {"h": 1, "l": 2}

# features of fire weather list to include
selected_weather_features = ['temp', 'rh', 'ws','wd_sin', 'wd_cos', 'prec', 'ffmc', 'dmc','dc', 'isi', 'bui']

def load_raster(path: str) -> np.ma.MaskedArray:
    """ Load raster from given path"""
    with rasterio.open(path) as src:
        raster = src.read(1, masked=True) # mask out the nodata (-9999 values)

    return raster
