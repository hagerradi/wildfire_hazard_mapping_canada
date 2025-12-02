"""
Module to generate fuel features required for ROS calculation. This calls an R script to run the FBP calculations and save the rasters.
"""
import os
import subprocess

import numpy as np
import rasterio

from data_preparation.feature_processing.weather import load_weather_list
from data_preparation.grid_loader.utils import load_raster, visualize_ignition_grid
from data_preparation.grid_loader.weather import weather_list_to_grid


def load_weather_grid(weather_list_file_path:str, zone_grid_file_path: str, season: int = 1):
    """Single function to run the weather grid creation"""
    selected_weather_features = ['ffmc', 'bui', 'ws']
    weather_csv = load_weather_list(weather_list_file_path, season = season)
    with rasterio.open(zone_grid_file_path) as src:
        data = src.read(1, masked=True)
        profile = src.profile
    out = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling = "mean")
    
    print(np.unique(out[:, :, 0], return_counts=True))
    print(np.unique(out[:, :, 1], return_counts=True))
    print(np.unique(out[:, :, 2], return_counts=True))

    with rasterio.open("data/fuel_data_temporary/ffmc.asc", "w", **profile) as dst:
        dst.write(out[:, :, 0], 1)
    
    with rasterio.open("data/fuel_data_temporary/bui.asc", "w", **profile) as dst:
        dst.write(out[:, :, 1], 1)
    
    with rasterio.open("data/fuel_data_temporary/ws.asc", "w", **profile) as dst:
        dst.write(out[:, :, 2], 1)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/fuel_data_temporary", exist_ok=True)

    # 1: generate weather features
    load_weather_grid(weather_list_file_path="../yan_bp3/hex05/burning_conditions_module/hex_05_weather_list.csv",
                     zone_grid_file_path="../yan_bp3/hex05/mapped_inputs/cfrs.asc")
    
    # 2: run the R script (fbp_features.R) externally to generate ROS
    # subprocess.run(["Rscript", "data_preparation/feature_processing/fuel_features/fbp_features.R"], check=True)
    # 3: visualize outputs
    ros_output = load_raster("data/fuel_data_temporary/ws.asc")
    print(np.unique_counts(ros_output))
    visualize_ignition_grid(ros_output, cause=1, season=1)