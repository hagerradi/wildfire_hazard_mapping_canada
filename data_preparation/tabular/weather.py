import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preparation.tabular.utils import check_weather_list

_MIN_MAX_COLS = ["RelativeHumidity", "FineFuelMoistureCode"]
_Z_SCORE_COLS = [
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


def save_weather_normalization_params(params: dict, path: str | Path) -> None:
    """Save weather normalization parameters to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        "min_max": {
            "cols": params["min_max"]["cols"],
            "min": [float(v) for v in params["min_max"]["min"]],
            "max": [float(v) for v in params["min_max"]["max"]],
        },
        "z_score": {
            "cols": params["z_score"]["cols"],
            "mean": [float(v) for v in params["z_score"]["mean"]],
            "std": [float(v) for v in params["z_score"]["std"]],
        },
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)


def load_weather_normalization_params(path: str | Path) -> dict:
    """Load weather normalization parameters from a JSON file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Weather normalization params not found: {path}")
    with open(path) as f:
        raw = json.load(f)
    return {
        "min_max": {
            "cols": raw["min_max"]["cols"],
            "min": np.array(raw["min_max"]["min"], dtype=np.float64),
            "max": np.array(raw["min_max"]["max"], dtype=np.float64),
        },
        "z_score": {
            "cols": raw["z_score"]["cols"],
            "mean": np.array(raw["z_score"]["mean"], dtype=np.float64),
            "std": np.array(raw["z_score"]["std"], dtype=np.float64),
        },
    }


def _add_wind_components(weather_list: pd.DataFrame) -> pd.DataFrame:
    """Add wind_x and wind_y columns computed from WindSpeed and WindDirection."""
    wind_x, wind_y = wind_to_components(
        ws=np.array(weather_list["WindSpeed"]),
        wd=np.array(weather_list["WindDirection"]),
    )
    weather_list = weather_list.copy()
    weather_list["wind_x"] = wind_x
    weather_list["wind_y"] = wind_y
    weather_list["Precipitation"] = np.log1p(weather_list["Precipitation"])
    return weather_list


def preprocess_weather_list(
    weather_list: pd.DataFrame,
    fit_mask: pd.Series | np.ndarray | None = None,
    norm_params_path: str | Path | None = None,
) -> pd.DataFrame:
    """Preprocess/Normalize the Fire Weather List.

    Parameters
    ----------
    weather_list:
        Raw weather DataFrame (must contain the expected weather columns).
    fit_mask:
        Boolean mask selecting rows to fit scalers on (train rows only).
        Ignored when ``norm_params_path`` is provided.
    norm_params_path:
        If given and the file exists, load normalization parameters from it
        instead of fitting new scalers (inference mode).
        If given and the file does not exist, fit scalers and save parameters
        to this path for future reuse (first-run / training mode).
    """
    weather_list = _add_wind_components(weather_list)

    if norm_params_path is not None and Path(norm_params_path).exists():
        # ── Apply pre-fitted parameters (inference / reuse mode) ──────────────
        params = load_weather_normalization_params(norm_params_path)

        mm_cols = params["min_max"]["cols"]
        mm_min = params["min_max"]["min"]
        mm_max = params["min_max"]["max"]
        scale = mm_max - mm_min
        scale[scale == 0] = 1.0
        weather_list[mm_cols] = (weather_list[mm_cols].to_numpy() - mm_min) / scale

        z_cols = params["z_score"]["cols"]
        z_mean = params["z_score"]["mean"]
        z_std = params["z_score"]["std"]
        z_std[z_std == 0] = 1.0
        weather_list[z_cols] = (weather_list[z_cols].to_numpy() - z_mean) / z_std

        return weather_list

    # ── Fit new scalers ───────────────────────────────────────────────────────
    if fit_mask is None:
        fit_mask_arr = np.ones(len(weather_list), dtype=bool)
    else:
        fit_mask_arr = np.asarray(fit_mask, dtype=bool)
        if fit_mask_arr.shape != (len(weather_list),):
            raise ValueError("fit_mask must have one boolean value per weather row.")
        if not fit_mask_arr.any():
            raise ValueError("fit_mask contains no rows; cannot fit weather preprocessing scalers.")

    scaler_mm = MinMaxScaler()
    scaler_mm.fit(weather_list.loc[fit_mask_arr, _MIN_MAX_COLS])
    weather_list[_MIN_MAX_COLS] = scaler_mm.transform(weather_list[_MIN_MAX_COLS])

    scaler_z = StandardScaler()
    scaler_z.fit(weather_list.loc[fit_mask_arr, _Z_SCORE_COLS])
    weather_list[_Z_SCORE_COLS] = scaler_z.transform(weather_list[_Z_SCORE_COLS])

    if norm_params_path is not None:
        params = {
            "min_max": {
                "cols": _MIN_MAX_COLS,
                "min": scaler_mm.data_min_,
                "max": scaler_mm.data_max_,
            },
            "z_score": {
                "cols": _Z_SCORE_COLS,
                "mean": scaler_z.mean_,
                "std": scaler_z.scale_,
            },
        }
        save_weather_normalization_params(params, norm_params_path)

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
