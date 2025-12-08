import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.plot import show

# value for nodata in the rasters
NODATA = np.nan

# normalization values for elevation (on national scale)
ELEV_NATIONAL_MAX = 5855
ELEV_NATIONAL_MIN = -158

# Max wind velocity (TODO: need to modify when we have the entire dataset)
MAX_WIND_VELOCITY = 14.279999732971191

# Normalization values for Fire Intensity (TODO: need to rerun once we have the entire dataset)
FIRE_INTENSITY_MAX = 127247.0
FIRE_INTENSITY_MIN = 0.0

# mapping cause to cause index
fire_cause_mapping = {1: "h", 2: "l"}

# features of fire weather list to include
selected_weather_features = ['temp', 'rh', 'prec', 'ffmc', 'dmc', 'dc', 'isi', 'bui'] #'ws','wd_sin', 'wd_cos'

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
            raster = src.read(1, masked=True) # mask out the nodata
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
    
def load_fire_shapefiles(hex_dir: str) -> list[Path]:
    """
    Helper that loads all shp files from a given hexel.

    Args:
        hex_dir (str): The path to the hexel folder
    Returns:
        (list[Path]): the list of shapefiles for the hexel
    """
    output_dir = Path(hex_dir) / "outputs"
    if not output_dir.exists():
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    shp_paths = sorted(list(output_dir.glob("*.shp")))
    if not shp_paths:
            raise FileNotFoundError(f"No .shp files found in {output_dir}")
    
    return shp_paths

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


def visualize_ignition_grid(grid: np.array, cause: int, season: int):
    """
    Visualizes an ignition raster using matplotlib.
    """
    masked_grid = np.ma.masked_invalid(grid)

    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        masked_grid,
        cmap="gray",
        origin="upper"
    )
    plt.colorbar(img, label="Ignition probability")
    plt.title(f"Ignition Grid for season {season} : Cause: {cause}")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
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
    except:  # noqa: E722
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


def visualize_fuel_grid(fuel_grid: np.array):
    """Visualize fuel grid"""
    classes = np.unique(fuel_grid)[:-1] # ignore the noData class

    # Reapply mask: -1 = nodata
    masked_grid = np.ma.masked_invalid(fuel_grid)

    colors = plt.cm.tab20(np.linspace(0, 1, len(classes)))  # or any discrete cmap

    cmap = ListedColormap(colors)

    plt.figure(figsize=(8, 6))
    plt.imshow(
        masked_grid,
        cmap=cmap,
        origin="upper",
        vmin=0,
        vmax=len(classes) - 1
    )

    plt.title("FBP fuel map")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")

    # Classification legend instead of colorbar
    legend_patches = [
        Patch(facecolor=colors[i], edgecolor="black", label=str(cls))
        for i, cls in enumerate(classes)
    ]
    plt.legend(
        handles=legend_patches,
        title="Fuel Classes",
        loc="upper right",
        bbox_to_anchor=(1.32, 1.0),
        frameon=True
    )

    plt.tight_layout()
    plt.show()


def visualize_elevation_grid(grid: np.ndarray):
    """
    Visualizes an elevation grid using matplotlib.
    """
    masked_grid = np.ma.masked_invalid(grid)

    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        masked_grid,
        cmap="terrain",
        origin="upper"
    )
    plt.colorbar(img, label="Elevation")
    plt.title("Elevation Grid (m)")
    plt.show()

def visualize_fire_intensity_grid(grid: np.ndarray):
    """
    Visualizes a fire intensity grid using matplotlib.
    """
    masked_grid = np.ma.masked_invalid(grid)

    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        masked_grid,
        cmap="viridis",
        origin="upper"
    )
    plt.colorbar(img, label="Fire Intensity")
    plt.title("Fire Intensity Grid (m)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.show()

def visualize_burn_prob_grid(grid: np.ndarray):
    """
    Visualizes a burn prob grid using matplotlib.
    """

    plt.figure(figsize=(8, 6))
    img = plt.imshow(
        grid,
        cmap="viridis",
        origin="upper"
    )
    plt.colorbar(img, label="Burn Probability")
    plt.title("Burn Probability Grid (m)")
    plt.xlabel("Easting (m)")
    plt.ylabel("Northing (m)")
    plt.show()