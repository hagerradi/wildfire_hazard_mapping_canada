import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional

import rasterio
from rasterio.plot import plotting_extent

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler

def wind_direction_to_sincos(wd:np.array)-> tuple[np.array, np.array]:
    """Convert Wind Direction to Sin/Cos for normalization"""
    # Since wd is circular, we cannot directly normalize so need to use it as sin/cos
    wd_deg = wd.reshape(-1)
    sin = np.sin(np.deg2rad(wd_deg))
    cos = np.cos(np.deg2rad(wd_deg))
    return sin,cos

def preprocess_weather_list(weather_list:pd.DataFrame)->pd.DataFrame:
    """Preprocess/Normalize the Fire Weather List"""
    #prec usually follows power law and 
    weather_list['prec'] = np.log1p(weather_list['prec'])  # log1p is log(x+1)

    # These are bounded variables
    min_max_cols = ['rh', 'ffmc']
    scaler_mm = MinMaxScaler()
    weather_list[min_max_cols] = scaler_mm.fit_transform(weather_list[min_max_cols])

    # These are normal dist variables
    z_score_cols = ['temp', 'ws', 'dmc', 'dc', 'isi', 'bui']
    scaler_z = StandardScaler()
    weather_list[z_score_cols] = scaler_z.fit_transform(weather_list[z_score_cols])
    
    wd_sin, wd_cos = wind_direction_to_sincos(np.array(weather_list["wd"]))
    weather_list["wd_sin"], weather_list["wd_cos"] = wd_sin, wd_cos
    # weather_list.drop(columns=['wd'], inplace=True)
    return weather_list

def load_weather_list(weather_list_file_path:str, season: Optional[int] = None)->pd.DataFrame:
    """Load and preprocess the weather list csv"""
    path = Path(weather_list_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather list file not found: {path}")
    weather_list = pd.read_csv(weather_list_file_path)
    if season:
        weather_list_subset = weather_list[weather_list["season"]==season].copy()
    else:
        weather_list_subset = weather_list.copy()
    weather_list_preprocessed = preprocess_weather_list(weather_list_subset)
    return weather_list_preprocessed

def weather_list_to_grid(weather_list:pd.DataFrame, fire_weather_zone_grid: np.ma.MaskedArray, sampling: str="random")->np.array:
    """Project weather list onto the Fire Weather Zones"""

    fwz = weather_list["wx_zone"].unique()
    selected_features = ['temp', 'rh', 'ws','wd_sin', 'wd_cos', 'prec', 'ffmc', 'dmc','dc', 'isi', 'bui']
    
    if sampling=="dist":
        out = np.repeat(fire_weather_zone_grid.data[..., np.newaxis], (len(selected_features))*2, axis=-1).astype("float32")
    else:
        out = np.repeat(fire_weather_zone_grid.data[..., np.newaxis], len(selected_features), axis=-1).astype("float32")
    
    for zone in fwz:
        weather_zone_subset = weather_list[weather_list["wx_zone"]==zone][selected_features]
        if sampling=="random":
            value = np.array(weather_zone_subset.sample(1, random_state=42) )
        elif sampling=="dist":
            value = np.vstack([weather_zone_subset.mean(), weather_zone_subset.std()]).T.flatten()
        else:
            print("NO WEATHER SAMPLING SELECTED; USING RANDOM SAMPLING")
            value = np.array(weather_zone_subset.sample(1, random_state=42))
        
        out[fire_weather_zone_grid.data==zone] = value
    return out

if __name__=="__main__":
    data_folder = "../yan_bp3/hex05"
    folders = ["burning_conditions_module", "dictionary", "ignitions_module",
             "mapped_inputs",  "outputs"]
    season = 1
    weather_csv = load_weather_list(data_folder + "/" + folders[0] + "/" + "hex_05_weather_list.csv", season)
