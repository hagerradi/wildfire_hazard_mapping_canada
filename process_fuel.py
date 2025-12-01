import numpy as np
import rasterio

from data_preparation.feature_processing.weather import load_weather_list
from data_preparation.grid_loader.utils import load_raster
from data_preparation.grid_loader.weather import weather_list_to_grid


def convert_elevation_to_slope(elevation_raster_path: str, output_slope_raster_path: str):
    """Convert elevation raster to slope raster and save to output path"""
    with rasterio.open(elevation_raster_path) as src:
        dem = src.read(1)
        transform = src.transform
        profile = src.profile
        res_x = transform.a
        res_y = -transform.e

    # Compute gradient
    dy, dx = np.gradient(dem, res_y, res_x)   # row, col
    slope_rad = np.arctan(np.sqrt(dx*dx + dy*dy))
    slope_deg = np.degrees(slope_rad)

    print(np.unique_counts(slope_deg))
    # Save slope raster
    profile.update(dtype="float32")

    with rasterio.open(output_slope_raster_path, "w", **profile) as dst:
        dst.write(slope_deg.astype("float32"), 1)


def load_weather_grid(weather_list_file_path:str, zone_grid_file_path: str):
    """Single function to run the weather grid creation"""
    selected_weather_features = ['ffmc', 'bui', 'ws']
    weather_csv = load_weather_list(weather_list_file_path, season = 1)
    data = load_raster(zone_grid_file_path)
    out = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling = "mean")
    
    with rasterio.open(zone_grid_file_path) as src:
        data = src.read(1)
        profile = src.profile
    
    with rasterio.open("example_inputs/ffmc.asc", "w", **profile) as dst:
        dst.write(out[:, :, 0], 1)
    
    with rasterio.open("example_inputs/bui.asc", "w", **profile) as dst:
        dst.write(out[:, :, 1], 1)
    
    with rasterio.open("example_inputs/ws.asc", "w", **profile) as dst:
        dst.write(out[:, :, 2], 1)


if __name__ == "__main__":
    # convert_elevation_to_slope(elevation_raster_path="../yan_bp3/hex05/mapped_inputs/elev.asc",
    #                            output_slope_raster_path="example_inputs/slope.asc")
    
    load_weather_grid(weather_list_file_path="../yan_bp3/hex05/burning_conditions_module/hex_05_weather_list.csv",
                     zone_grid_file_path="../yan_bp3/hex05/mapped_inputs/cfrs.asc")