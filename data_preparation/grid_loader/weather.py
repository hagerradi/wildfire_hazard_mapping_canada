import numpy as np
import pandas as pd

from data_preparation.feature_processing.weather import load_weather_list
from data_preparation.grid_loader.utils import NODATA, load_raster, selected_weather_features, visualize_weather_params


def weather_list_to_grid(
    weather_list: pd.DataFrame, fire_weather_zone_grid: np.ma.MaskedArray, selected_weather_features: list[str], sampling: str = "dist"
) -> np.ndarray:
    """Project weather list onto the Fire Weather Zones"""
    fire_weather_zones = weather_list["wx_zone"].unique()

    if sampling == "dist":
        out = np.repeat(fire_weather_zone_grid[..., np.newaxis], (len(selected_weather_features)) * 2, axis=-1).astype("float32")
    elif sampling == "weather_zone_id":
        out = np.repeat(fire_weather_zone_grid[..., np.newaxis], 1, axis=-1).astype("float32")
    else:
        out = np.repeat(fire_weather_zone_grid[..., np.newaxis], len(selected_weather_features), axis=-1).astype("float32")

    for zone in fire_weather_zones:
        weather_zone_subset = weather_list[weather_list["wx_zone"] == zone][selected_weather_features]
        # Randomly sample 1 row from the given weather list (1xlen(selected_weather_features))
        if sampling == "random":
            value = np.array(weather_zone_subset.sample(1, random_state=42))
        # array of means for all the variables (1xlen(selected_weather_features))
        elif sampling == "mean":
            value = weather_zone_subset.mean().values
        # array of mean, var for all the variables (1x2*len(selected_weather_features))
        elif sampling == "dist":
            value = np.vstack([weather_zone_subset.mean(), weather_zone_subset.std()]).T.flatten()
        # array of weather zone id 
        elif sampling == "weather_zone_id":
            value = zone
        out[fire_weather_zone_grid.data == zone] = value  # type: ignore
    # HxWx2*len(selected_weather_features) (or len(selected_weather_features))
    return out.filled(NODATA)  # type: ignore


def load_weather_grid(weather_list_file_path: str, zone_grid_file_path: str, season: int = None, sampling: str = "weather_zone_id"): # TODO dirty change for now, expose this sampling as an arg
    """Single function to run the weather grid creation"""
    weather_csv = load_weather_list(weather_list_file_path, season)
    data = load_raster(zone_grid_file_path)
    out = weather_list_to_grid(weather_csv, data, selected_weather_features, sampling)
    return out


# TODO: DELETE LATER
if __name__ == "__main__":
    data_folder = "../yan_bp3/hex05"
    folders = ["burning_conditions_module", "dictionary", "ignitions_module", "mapped_inputs", "outputs"]
    sampling = "dist"
    out = load_weather_grid(
        weather_list_file_path=data_folder + "/" + folders[0] + "/" + "hex_05_weather_list.csv",
        zone_grid_file_path=data_folder + "/" + folders[3] + "/cfrs.asc",
        season=1,
        sampling=sampling,
    )

    print(out.shape)
    print(np.unique_counts(out))
    visualize_weather_params(out, sampling=sampling)
