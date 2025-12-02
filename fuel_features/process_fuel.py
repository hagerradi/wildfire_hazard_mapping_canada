import os

import numpy as np
import rasterio

from data_preparation.feature_processing.weather import load_weather_list
from data_preparation.grid_loader.utils import load_raster, visualize_ignition_grid
from data_preparation.grid_loader.weather import weather_list_to_grid


def load_weather_grid(weather_list_file_path:str, zone_grid_file_path: str):
    """Single function to run the weather grid creation"""
    selected_weather_features = ['ffmc', 'bui', 'ws']
    weather_csv = load_weather_list(weather_list_file_path, season = 1)
    data = load_raster(zone_grid_file_path)
    out = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling = "mean")
    
    with rasterio.open(zone_grid_file_path) as src:
        data = src.read(1)
        profile = src.profile
    
    os.makedirs("data", exist_ok=True)
    os.makedirs("fuel_data_temporary", exist_ok=True)
    with rasterio.open("fuel_data_temporary/ffmc.asc", "w", **profile) as dst:
        dst.write(out[:, :, 0], 1)
    
    with rasterio.open("fuel_data_temporary/bui.asc", "w", **profile) as dst:
        dst.write(out[:, :, 1], 1)
    
    with rasterio.open("fuel_data_temporary/ws.asc", "w", **profile) as dst:
        dst.write(out[:, :, 2], 1)


if __name__ == "__main__":
    # load_weather_grid(weather_list_file_path="../yan_bp3/hex05/burning_conditions_module/hex_05_weather_list.csv",
    #                  zone_grid_file_path="../yan_bp3/hex05/mapped_inputs/cfrs.asc")

    fbp_input = load_raster("fuel_data_temporary/fbp.asc")
    print(np.unique_counts(fbp_input))

    ros_output = load_raster("fuel_data_temporary/ROS.asc")
    print(np.unique_counts(ros_output))

    visualize_ignition_grid(ros_output, cause=1, season=1)