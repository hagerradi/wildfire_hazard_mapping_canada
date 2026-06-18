from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preparation.tabular.utils import check_weather_list


def wind_direction_to_sincos(wd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert wind direction (degrees) to sin/cos components."""
    wd_rad = np.deg2rad(wd.reshape(-1))
    return np.sin(wd_rad), np.cos(wd_rad)


def wind_to_components(ws: np.ndarray, wd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert wind speed and direction to Cartesian vector components.

    Encodes magnitude and direction jointly so that averaging the components
    yields the physically correct resultant wind vector — unlike averaging
    speed and direction independently.

    Returns:
        wind_x: east-west component  (WindSpeed * sin(WindDirection))
        wind_y: north-south component (WindSpeed * cos(WindDirection))
    """
    wd_rad = np.deg2rad(wd.reshape(-1))
    wind_x = ws.reshape(-1) * np.sin(wd_rad)
    wind_y = ws.reshape(-1) * np.cos(wd_rad)
    return wind_x, wind_y


def preprocess_weather_list(weather_list: pd.DataFrame, fit_mask: pd.Series | np.ndarray | None = None) -> pd.DataFrame:
    """Preprocess/Normalize the Fire Weather List"""
    weather_list = weather_list.copy()
    if fit_mask is None:
        fit_mask_arr = np.ones(len(weather_list), dtype=bool)
    else:
        fit_mask_arr = np.asarray(fit_mask, dtype=bool)
        if fit_mask_arr.shape != (len(weather_list),):
            raise ValueError("fit_mask must have one boolean value per weather row.")
        if not fit_mask_arr.any():
            raise ValueError("fit_mask contains no rows; cannot fit weather preprocessing scalers.")

    # Compute wind vector components from raw values before any normalization
    wind_x, wind_y = wind_to_components(
        ws=np.array(weather_list["WindSpeed"]),
        wd=np.array(weather_list["WindDirection"]),
    )
    weather_list["wind_x"] = wind_x
    weather_list["wind_y"] = wind_y

    # Precipitation follows a power-law distribution
    weather_list["Precipitation"] = np.log1p(weather_list["Precipitation"])

    # Bounded variables → min-max scaling
    min_max_cols = ["RelativeHumidity", "FineFuelMoistureCode"]
    scaler_mm = MinMaxScaler()
    scaler_mm.fit(weather_list.loc[fit_mask_arr, min_max_cols])
    weather_list[min_max_cols] = scaler_mm.transform(weather_list[min_max_cols])

    # Approximately normal variables → z-score normalization
    # wind_x and wind_y are included here: they share the scale of WindSpeed
    z_score_cols = [
        "Temperature",
        "WindSpeed",
        "DuffMoistureCode",
        "DroughtCode",
        "InitialSpreadIndex",
        "BuildupIndex",
        "FireWeatherIndex",
        "wind_x",
        "wind_y",
    ]
    scaler_z = StandardScaler()
    scaler_z.fit(weather_list.loc[fit_mask_arr, z_score_cols])
    weather_list[z_score_cols] = scaler_z.transform(weather_list[z_score_cols])

    wd_sin, wd_cos = wind_direction_to_sincos(np.array(weather_list["WindDirection"]))
    weather_list["wd_sin"], weather_list["wd_cos"] = wd_sin, wd_cos
    return weather_list


def load_weather_list(weather_list_file_path: str, season: int | None = None, normalize_weatherlist: bool = True) -> pd.DataFrame:
    """Load and preprocess the weather list csv"""
    path = Path(weather_list_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather list file not found: {path}")
    weather_list = pd.read_csv(weather_list_file_path)
    weather_list = check_weather_list(weather_list)
    weather_list = weather_list.loc[
        :, ~weather_list.columns.str.startswith("Unnamed:")
    ]  # Accounts for anomaly columns in hex 32 (just a repeat column of WeatherZone)
    weather_list_subset = weather_list[weather_list["Season"] == season].copy() if season else weather_list.copy()

    if normalize_weatherlist:
        return preprocess_weather_list(weather_list_subset)

    return weather_list_subset
