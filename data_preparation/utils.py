import math
import numpy as np

import rasterio
from rasterio.plot import show

from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

fire_cause_mapping = {"h": 1, "l": 2}

def load_raster(path: str) -> np.ma.MaskedArray:
    """ Load raster from given path"""
    with rasterio.open(path) as src:
        raster = src.read(1, masked=True) # mask out the nodata (-9999 values)

    return raster

def visualize_ignition_raster(raster: np.array, cause: int, season: int):
    """
    Visualizes an ignition raster using matplotlib.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    show(raster, ax=ax, cmap="gray")

    img = ax.get_images()[0]
    cbar = plt.colorbar(img, ax=ax, cmap='gray')
    cbar.set_label("ignition value")

    ax.set_title(f"Ignition Grid for season {season} : Cause: {cause}")
    plt.show()

def visualize_weather_params(weather_cube: np.array, sampling:str="dist", cols:int=3):
    """
    Visualizes multiple weather parameters in a subplot grid.
    
    Args:
        weather_cube: 3D numpy array (Height, Width, Num_Params)
        param_names: List of strings matching the 3rd dimension of weather_cube
        cols: Number of columns desired in the grid
    """
    if sampling=="dist":
        param_names = ['temp_mean','temp_std',
        'rh_mean', 'rh_std', 
        'ws_mean','ws_std',
        'wd_sin_mean','wd_sin_std', 
        'wd_cos_mean', 'wd_cos_std',
        'prec_mean', 'prec_std',
        'ffmc_mean', 'ffmc_std', 
        'dmc_mean','dmc_std',
        'dc_mean', 'dc_std', 
        'isi_mean', 'isi_std', 
        'bui_mean','bui_std']
    else:
        param_names = ['temp', 'rh', 'ws','wd_sin', 'wd_cos', 'prec', 'ffmc', 'dmc','dc', 'isi', 'bui']
    num_params = len(param_names)
    
    # 1. Calculate Grid Size
    rows = math.ceil(num_params / cols)
    
    # 2. Create Subplots
    # Increase figsize to accommodate multiple plots (width, height)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
    
    # Flatten axes array for easy iteration (handles 1D or 2D arrays of axes)
    axes = axes.flatten() 

    # 3. Define Colormap (Doing this once usually suffices)
    # Note: cm.get_cmap is deprecated in recent versions; using recommended approach fallback
    try:
        base_cmap = plt.get_cmap("viridis").copy()
    except:
        base_cmap = cm.get_cmap("viridis", 256).copy()
        
    base_cmap.set_bad(color=(0, 0, 0, 0)) # Transparent for masked values

    # 4. Loop through parameters
    for i, param in enumerate(param_names):
        ax = axes[i]
        
        # Extract single slice: (Height, Width)
        raster_slice = weather_cube[:, :, i]
        
        # Mask the data
        masked = np.ma.masked_equal(raster_slice, -9999)

        # Plot using rasterio.plot.show
        # We pass the ax object so it draws on the specific subplot
        show(masked, ax=ax, cmap=base_cmap, interpolation="nearest")
        
        # Add Colorbar specifically to this axis
        img = ax.get_images()[0] 
        cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(param)
        
        # Set Title
        ax.set_title(f"{param}")

    # 5. Hide empty subplots (if you have 7 params in a 3x3 grid, hide last 2)
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.show()