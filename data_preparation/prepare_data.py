import numpy as np


def split_hexel(stacked_feats:np.ndarray, mask:np.ndarray, win_h:int=128, win_w:int=128, overlap_ratio:float=0.2, mask_threshold:float=0.5)->tuple[list[np.ndarray], list[list]]:
    """Split the hexel using sliding windows for inp to the model"""
    H,W,C = stacked_feats.shape
    stride_h = max(1,np.floor(win_h * (1 - overlap_ratio))) #n_rows = (H-win_h)//stride_h + 1
    stride_w = max(1,np.floor(win_w * (1 - overlap_ratio)))
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
    return valid_windows, valid_coords #(n_rows*n_cols,win_h,win_w), (n_rows*n_cols,3)