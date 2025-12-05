"""
Module to generate fuel features required for ROS calculation. This calls an R script to run the FBP calculations and save the rasters.
"""
import copy
import os
import subprocess

import numpy as np
import rasterio

from data_preparation.feature_processing.weather import load_weather_list
from data_preparation.grid_loader.utils import load_raster, visualize_ignition_grid
from data_preparation.grid_loader.weather import weather_list_to_grid


def write_grid(path: str, arr2d: np.ndarray, profile: dict):
    # Ensure nodata in profile matches what we use
    profile = profile.copy()
    # If it's a masked array, fill with nodata
    if np.ma.isMaskedArray(arr2d):  # noqa: SIM108
        arr2d = arr2d.filled(profile["nodata"])
    else:
        # Replace NaNs with nodata
        arr2d = np.where(np.isnan(arr2d), profile["nodata"], arr2d)

    arr2d = arr2d.astype("float32")

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr2d, 1)

def load_weather_grid(weather_list_file_path:str, zone_grid_file_path: str):
    """Single function to run the weather grid creation"""
    selected_weather_features = ['ffmc', 'bui', 'ws']
    weather_csv = load_weather_list(weather_list_file_path, season = 1, normalize_weatherlist=False)

    with rasterio.open(zone_grid_file_path) as src:
        data = src.read(1, masked=True)
        profile = src.profile

    out_weather_params = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling = "mean")
    weather_profile = copy.deepcopy(profile)
    weather_profile.update(
        dtype="float32",
        height=out_weather_params.shape[0],
        width=out_weather_params.shape[1],
        count=1,
        # to unify with original fbp and elev - not unified with our NODATA
        nodata=-9999.0,
    )
    write_grid("data/fuel_input_data/ffmc.asc", out_weather_params[:, :, 0], weather_profile)
    write_grid("data/fuel_input_data/bui.asc",  out_weather_params[:, :, 1], weather_profile)
    write_grid("data/fuel_input_data/ws.asc",   out_weather_params[:, :, 2], weather_profile)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/fuel_input_data", exist_ok=True)

    use_constant_FWI = False # this has to be the same as in the R script
    if not use_constant_FWI:
        # 1: generate weather features
        load_weather_grid(weather_list_file_path="../yan_bp3/hex05/burning_conditions_module/hex_05_weather_list.csv",
                        zone_grid_file_path="../yan_bp3/hex05/mapped_inputs/cfrs.asc")
    
    # # 2: run the R script (fbp_features.R) externally to generate ROS
    subprocess.run(["Rscript", "data_preparation/feature_processing/fuel_features/fbp_features.R"], check=True)
    
    # # 3: visualize outputs
    ros_output = load_raster("data/fuel_input_data/ROS.asc")
    print(np.unique_counts(ros_output))
    visualize_ignition_grid(ros_output, cause=1, season=1)