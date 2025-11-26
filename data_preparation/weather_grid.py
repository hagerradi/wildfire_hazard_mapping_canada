from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from data_preparation.utils import load_raster, selected_weather_features
from data_preparation.visualize import visualize_weather_params


def wind_direction_to_sincos(wd:np.ndarray)-> tuple[np.ndarray, np.ndarray]:
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
    return weather_list

def load_weather_list(weather_list_file_path:str, season: int | None = None)->pd.DataFrame:
    """Load and preprocess the weather list csv"""
    path = Path(weather_list_file_path)
    if not path.exists():
        raise FileNotFoundError(f"Weather list file not found: {path}")
    weather_list = pd.read_csv(weather_list_file_path)
    weather_list_subset = weather_list[weather_list["season"] == season].copy() if season else weather_list.copy()
    weather_list_preprocessed = preprocess_weather_list(weather_list_subset)
    return weather_list_preprocessed

def weather_list_to_grid(weather_list:pd.DataFrame, fire_weather_zone_grid: np.ma.MaskedArray, sampling: str="dist")->np.ndarray:
    """Project weather list onto the Fire Weather Zones"""

    fire_weather_zones = weather_list["wx_zone"].unique()
    
    if sampling=="dist":
        out = np.repeat(fire_weather_zone_grid.data[..., np.newaxis], (len(selected_weather_features))*2, axis=-1).astype("float32")
    else:
        out = np.repeat(fire_weather_zone_grid.data[..., np.newaxis], len(selected_weather_features), axis=-1).astype("float32")
    
    for zone in fire_weather_zones:
        weather_zone_subset = weather_list[weather_list["wx_zone"]==zone][selected_weather_features]
        #Randomly sample 1 row from the given weather list (1xlen(selected_weather_features))
        if sampling=="random":
            value = np.array(weather_zone_subset.sample(1, random_state=42) )
        #array of means for all the variables (1xlen(selected_weather_features))
        elif sampling=="mean":
            value = weather_zone_subset.mean().values
        #array of mean, var for all the variables (1x2*len(selected_weather_features))
        elif sampling=="dist":
            value = np.vstack([weather_zone_subset.mean(), weather_zone_subset.std()]).T.flatten()        
        out[fire_weather_zone_grid.data==zone] = value # type: ignore
    return out #HxWx2*len(selected_weather_features) (or len(selected_weather_features))

def build_weather_grid(weather_list_file_path:str, zone_grid_file_path: str, season: int=1, sampling:str="dist"):
    """Single function to run the weather grid creation"""
    weather_csv = load_weather_list(weather_list_file_path, season)
    data = load_raster(zone_grid_file_path)

    out = weather_list_to_grid(weather_csv, data, sampling)
    return out 

#TODO: DELETE LATER
if __name__=="__main__":
    data_folder = "../yan_bp3/hex05"
    folders = ["burning_conditions_module", "dictionary", "ignitions_module",
             "mapped_inputs",  "outputs"]
    sampling="dist"
    out = build_weather_grid(weather_list_file_path=data_folder + "/" + folders[0] + "/" + "hex_05_weather_list.csv",
                    zone_grid_file_path=data_folder + "/" + folders[3] + "/cfrs.asc", 
                    season=1,
                    sampling=sampling)

    print(out.shape)
    print(np.unique_counts(out))
    visualize_weather_params(out, sampling=sampling)
