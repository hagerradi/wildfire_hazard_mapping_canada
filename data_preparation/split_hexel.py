
import numpy as np

from data_preparation.load_hexel_data import load_features_per_hexel
from data_preparation.utils import plot_split_window_hexel


def get_split_hexel_window(stacked_feats:np.ndarray, mask:np.ndarray, win_h:int=128, win_w:int=128, overlap_ratio:float=0.2, mask_threshold:float=0.5)->tuple[list[np.ndarray], list[dict]]:
    """
        Split the hexel using sliding windows for inp to the model
        Inputs:
            stacked_feats(np.ndarray): Stacked array of all the features of the shape (H,W, num_feats)
            mask(np.ndarray): A bool array where True means to ignore the pixel (H,W)
    """
    H,W,_ = stacked_feats.shape
    stride_h = max(1,int(win_h * (1 - overlap_ratio))) #n_rows = (H-win_h)//stride_h + 1
    stride_w = max(1,int(win_w * (1 - overlap_ratio)))
    window_area = win_h * win_w
    num_total_windows, num_valid_windows = 0.0,0.0
    valid_windows, valid_coords = [], []

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
                valid_windows.append(window_data)
                valid_coords.append({(row, col): valid_ratio})
    #TODO: Saving function
    return valid_windows, valid_coords #(n_rows*n_cols,win_h,win_w), (n_rows*n_cols,3)

if __name__=="__main__":
    root_dir = "../yan_bp3"
    stacked_feats, mask = load_features_per_hexel(root_dir=root_dir, hex_id="05")
    
    valid_windows, valid_coords = get_split_hexel_window(stacked_feats=stacked_feats[0], 
                                        mask=mask[0], 
                                        win_h=128, 
                                        win_w=128, 
                                        overlap_ratio=0.2, 
                                        mask_threshold=0.5)
    print(len(valid_windows), valid_windows[0].shape)
    plot_split_window_hexel(valid_windows[:10], -1)