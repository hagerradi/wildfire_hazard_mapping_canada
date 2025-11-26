import os

import numpy as np

from data_preparation.elevation_grid import load_elevation_grid
from data_preparation.ignition_grid import load_ignition_grid
from data_preparation.output_bp_grid import load_output_burn_prob_grid
from data_preparation.weather_grid import build_weather_grid
from data_preparation.wind_grid import get_wind_grid_sample


def get_stacked_feats(root_dir:str, season:int, cause:int, wind_sampling:str, weather_sampling:str)->tuple[np.ndarray, np.ndarray]:
    """Stack all the features"""
    #TODO: Channel Mapping dict, Add Fuel grid features
    elevation_grid = load_elevation_grid(path=os.path.join(root_dir, "mapped_inputs/elev.asc"))
    ignition_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, "ignitions_module/ignition_grids"), season=season, cause=cause)
    out_bp_grid = load_output_burn_prob_grid(path=os.path.join(root_dir, "outputs/hex_05_20000_iter_bp.tif"))  # TODO: Make the filepath dynamic
    weather_grid = build_weather_grid(weather_list_file_path=os.path.join(root_dir, "burning_conditions_module/hex_05_weather_list.csv"),
                        zone_grid_file_path=os.path.join(root_dir, "mapped_inputs/cfrs.asc"), 
                        season=season,
                        sampling=weather_sampling)
    wind_grid = get_wind_grid_sample(os.path.join(root_dir, "burning_conditions_module/wind_grids"),sampling=wind_sampling)
    print(elevation_grid.shape, ignition_grid.shape, weather_grid.shape, wind_grid.shape, out_bp_grid.shape)
    stacked_feats = np.concatenate([elevation_grid[:, :, np.newaxis], ignition_grid[:, :, np.newaxis], weather_grid, wind_grid, out_bp_grid[:, :, np.newaxis]],axis=-1)
    return stacked_feats, np.isnan(elevation_grid) #(H,W,35), (H,W)

def split_hexel(stacked_feats:np.ndarray, mask:np.ndarray, win_h:int=128, win_w:int=128, overlap_ratio:float=0.2, mask_threshold:float=0.5)->tuple[list[np.ndarray], list[list]]:
    """Split the hexel using sliding windows for inp to the model"""
    H,W,C = stacked_feats.shape
    stride_h = max(1,int(win_h * (1 - overlap_ratio))) #n_rows = (H-win_h)//stride_h + 1
    stride_w = max(1,int(win_w * (1 - overlap_ratio)))
    window_area = win_h * win_w
    total, processed = 0.0,0.0
    valid_windows, valid_coords = [], []

    for r in range(0, H - win_h + 1, stride_h):
        for c in range(0, W - win_w + 1, stride_w):
            total+=1
            # Extract the mask patch 
            mask_patch = mask[r : r + win_h, c : c + win_w]
            # Count True values in the mask
            true_count = np.count_nonzero(mask_patch)
            ratio = true_count / window_area
            # Check Threshold Condition
            if ratio <= mask_threshold:
                processed+=1
                window_data = stacked_feats[r : r + win_h, c : c + win_w, :]
                valid_windows.append(window_data)
                valid_coords.append((r, c, ratio))
    #TODO: Saving function
    return valid_windows, valid_coords #(n_rows*n_cols,win_h,win_w), (n_rows*n_cols,3)

if __name__=="__main__":
    root_dir = "../yan_bp3/hex05"
    stacked_feats, mask = get_stacked_feats(root_dir=root_dir, 
                                      season=1, 
                                      cause=1, 
                                      wind_sampling="all", 
                                      weather_sampling="dist")
    valid_windows, valid_coords = split_hexel(stacked_feats=stacked_feats, 
                                        mask=mask, 
                                        win_h=128, 
                                        win_w=128, 
                                        overlap_ratio=0.2, 
                                        mask_threshold=0.5)
    print(len(valid_windows), valid_windows[0].shape)