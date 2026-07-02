"""
Module to generate fuel features required for ROS calculation. This calls an R script to run the FBP calculations and save the rasters.
"""

import copy
import os
import subprocess

import numpy as np
import pandas as pd
import rasterio

from data_preparation.spatial.utils import NODATA, load_raster
from data_preparation.tabular.weather import load_weather_list
from data_preparation.visualizations import visualize_ignition_grid


def weather_list_to_grid(
    weather_list: pd.DataFrame, fire_weather_zone_grid: np.ma.MaskedArray, selected_weather_features: list[str], sampling: str = "dist"
) -> np.ndarray:
    """Project weather list onto the Fire Weather Zones"""
    fire_weather_zones = weather_list["WeatherZone"].unique()

    # Determine number of output channels based on sampling strategy
    if sampling == "dist":
        n_channels = len(selected_weather_features) * 2
    elif sampling == "weather_zone_id":
        n_channels = 1
    else:
        n_channels = len(selected_weather_features)
    h, w = fire_weather_zone_grid.shape
    out = np.full((h, w, n_channels), NODATA, dtype="float32")

    for zone in fire_weather_zones:
        weather_zone_subset = weather_list[weather_list["WeatherZone"] == zone][selected_weather_features]
        # Randomly sample 1 row from the given weather list (1xlen(selected_weather_features))
        if sampling == "random":
            value = np.array(weather_zone_subset.sample(1, random_state=42))
        # array of means for all the variables (1xlen(selected_weather_features))
        elif sampling == "mean":
            value = weather_zone_subset.mean().values
        # array of mean, var for all the variables (1x2*len(selected_weather_features))
        elif sampling == "dist":
            value = np.vstack([weather_zone_subset.mean(), weather_zone_subset.std()]).T.flatten()
        elif sampling == "weather_zone_id":
            value = zone
        out[fire_weather_zone_grid.data == zone] = value  # type: ignore
    # HxWx2*len(selected_weather_features) (or len(selected_weather_features)) (or 1)
    return out  # type: ignore


def save_grid(path: str, data_arr: np.ndarray, profile: dict):
    """Save a single-band raster as GeoTIFF."""

    profile = profile.copy()

    profile.update(
        driver="GTiff",
        dtype="float32",
        count=1,
        nodata=profile.get("nodata", -9999.0),
        compress="lzw",
    )

    data_arr = np.asarray(data_arr, dtype="float32")

    nodata = profile["nodata"]
    data_arr = np.nan_to_num(data_arr, nan=nodata)

    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data_arr, 1)


def load_weather_grid(weather_list_file_path: str, zone_grid_file_path: str):
    """Single function to run the weather grid creation"""
    selected_weather_features = ["FineFuelMoistureCode", "BuildupIndex", "WindSpeed", "WindDirection"]
    weather_csv = load_weather_list(weather_list_file_path, season=1, normalize_weatherlist=False)

    with rasterio.open(zone_grid_file_path) as src:
        data = src.read(1, masked=True)
        profile = src.profile

    out_weather_params = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling="mean")  # noqa: F821
    weather_profile = copy.deepcopy(profile)
    weather_profile.update(
        dtype="float32",
        height=out_weather_params.shape[0],
        width=out_weather_params.shape[1],
        count=1,
        # to unify with original fbp and elev - not unified with our NODATA
        nodata=-9999.0,
    )
    save_grid("data/fuel_data/ffmc.tif", out_weather_params[:, :, 0], weather_profile)
    save_grid("data/fuel_data/bui.tif", out_weather_params[:, :, 1], weather_profile)
    save_grid("data/fuel_data/ws.tif", out_weather_params[:, :, 2], weather_profile)
    save_grid("data/fuel_data/wd.tif", out_weather_params[:, :, 3], weather_profile)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    os.makedirs("data/fuel_data", exist_ok=True)

    # 1: generate weather features
    # load_weather_grid(
    #     weather_list_file_path="../burnp3plus/hex05/tabular/hex05_DailyWeather.csv",
    #     zone_grid_file_path="../burnp3plus/hex05/spatial/hex05_firezones.tif",
    # )

    # 2: run the R script (fbp_features.R) externally to generate ROS
    subprocess.run(["Rscript", "data_preparation/tabular/fuel_features/compute_fbp_features_v2.R"], check=True)

    # 3: visualize outputs
    ros_output = load_raster("data/fuel_data/HFI.tif")
    print(np.unique(ros_output))
    visualize_ignition_grid(ros_output, cause=1, season=1)
