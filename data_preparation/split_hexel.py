import os

import numpy as np
import pandas as pd

from data_preparation.load_hexel_data import load_features_per_hexel


def save_split_hexel_windows(valid_window:np.ndarray, out_dir:str, win_id:int, season:str, cause:str, hex_id:str, format:str="npy")->str:
    """Save a hexel window"""
    filename = os.path.join(out_dir, f"numpy_files/hex_{hex_id}_{str(win_id)}_{season}_{cause}.{format}")
    np.save(filename, valid_window) if format=="npy" else np.savez_compressed(filename, arr=valid_window)
    return filename

def get_split_hexel_window(season_cause_stacked_feats:np.ndarray, season_cause_mask:np.ndarray, season_cause_mapping:dict|None, out_dir:str, hex_id:str, win_h:int=128, win_w:int=128, overlap_ratio:float=0.2, mask_threshold:float=0.5):
    """
        Split the hexel using sliding windows for inp to the model
        Inputs:
            season_cause_stacked_feats(np.ndarray): Stacked array of all the features of the shape (num_season_cause, H, W, num_feats)
            season_cause_mask(np.ndarray): A bool array where True means to ignore the pixel (num_season_cause, H,W)
    """
    print("============Splitting the hexel==================")
    num_season_cause,H,W,_ = season_cause_stacked_feats.shape
    stride_h = max(1,int(win_h * (1 - overlap_ratio))) #n_rows = (H-win_h)//stride_h + 1
    stride_w = max(1,int(win_w * (1 - overlap_ratio)))
    window_area = win_h * win_w
    num_total_windows, num_valid_windows = 0.0,0.0
    valid_coords = []
    for i in range(num_season_cause):
        stacked_feats = season_cause_stacked_feats[i]
        mask = season_cause_mask[i]
        if season_cause_mapping is None:
            season, cause = "all", "all"
        else:
            season, cause = season_cause_mapping[i]
        for row in range(0, H - win_h + 1, stride_h):
            for col in range(0, W - win_w + 1, stride_w):
                num_total_windows+=1
                # Extract the mask patch 
                mask_window = mask[row : row + win_h, col : col + win_w]
                # Count True values in the mask
                true_count = np.count_nonzero(~mask_window)
                valid_ratio = true_count / window_area
                # Check Threshold Condition
                if valid_ratio>=mask_threshold:
                    num_valid_windows+=1
                    window_data = stacked_feats[row : row + win_h, col : col + win_w, :]
                    filename = save_split_hexel_windows(window_data, out_dir, int(num_valid_windows), season, cause, hex_id)
                    valid_coords.append([filename, season, cause, hex_id, num_valid_windows, row, col, valid_ratio])
    df_coords = pd.DataFrame(valid_coords)
    df_coords.columns = ["filename", "season", "cause", "hex_id", "window_id", "row", "col", "valid_ratio"]
    df_coords.to_csv(os.path.join(out_dir, f"meta_hex_{hex_id}.csv"), index=False)
    print(f"=====Hexel data Saved at {out_dir} ========")

if __name__=="__main__":
    root_dir = "../yan_bp3"
    modelling_approach = 1
    out_dir=f"../yan_bp3/data_samples_approach_{modelling_approach}"
    stacked_feats, mask, season_cause_mapping = load_features_per_hexel(root_dir=root_dir, hex_id="05", modelling_approach=modelling_approach)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "numpy_files"), exist_ok=True)
    get_split_hexel_window(season_cause_stacked_feats=stacked_feats, 
                                        season_cause_mask=mask, 
                                        season_cause_mapping=season_cause_mapping, 
                                        out_dir=out_dir, 
                                        hex_id="05",
                                        win_h=128, 
                                        win_w=128, 
                                        overlap_ratio=0.2, 
                                        mask_threshold=0.5) 