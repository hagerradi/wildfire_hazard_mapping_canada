from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preparation.tabular.utils import check_weather_list


def wind_direction_to_sincos(wd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert Wind Direction to Sin/Cos for normalization"""
    # Since wd is circular, we cannot directly normalize so need to use it as sin/cos
    wd_deg = wd.reshape(-1)
    sin = np.sin(np.deg2rad(wd_deg))
    cos = np.cos(np.deg2rad(wd_deg))
    return sin, cos


def preprocess_weather_list(weather_list: pd.DataFrame) -> pd.DataFrame:
    """Preprocess/Normalize the Fire Weather List"""
    # prec usually follows power law and
    weather_list["Precipitation"] = np.log1p(weather_list["Precipitation"])  # log1p is log(x+1)

    # These are bounded variables
    min_max_cols = ["RelativeHumidity", "FineFuelMoistureCode"]
    scaler_mm = MinMaxScaler()
    weather_list[min_max_cols] = scaler_mm.fit_transform(weather_list[min_max_cols])

    # These are normal dist variables
    z_score_cols = ["Temperature", "WindSpeed", "DuffMoistureCode", "DroughtCode", "InitialSpreadIndex", "BuildupIndex"]
    scaler_z = StandardScaler()
    weather_list[z_score_cols] = scaler_z.fit_transform(weather_list[z_score_cols])

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
