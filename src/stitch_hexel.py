import os

import numpy as np
import pandas as pd


def stitch_windows(windows, coords, original_shape, mode="average"):
    """
    Reconstructs an image from overlapping windows using either averaging or maximization.

    Args:
        windows (list of np.array): List of window arrays (H_win, W_win, C).
        coords (list of tuples): List of (row, col) top-left coordinates for each window.
        original_shape (tuple): Shape of the target hexel (H, W, C).
        mode (str): 'average' to mean overlapping pixels, 'max' to take the maximum.

    Returns:
        np.array: The reconstructed image.
    """
    dtype = np.float64

    if mode == "average":
        # --- AVERAGE MODE ---
        accumulator = np.zeros(original_shape, dtype=dtype)
        counter = np.zeros(original_shape, dtype=dtype)

        for window, (r, c) in zip(windows, coords):
            h_win, w_win = window.shape[:2]

            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Accumulate Sum and Count
            accumulator[r:r_end, c:c_end, :] += window[:h_paste, :w_paste, :]
            counter[r:r_end, c:c_end, :] += 1.0

        # Normalize
        accumulator[accumulator == np.nan] = 0.0
        valid_mask = counter > 0
        reconstructed = np.zeros_like(accumulator)
        reconstructed[valid_mask] = accumulator[valid_mask] / counter[valid_mask]

        return reconstructed

    elif mode == "max":
        # --- MAX MODE ---
        # Initialize with negative infinity so any real data (even negative) will override it
        accumulator = np.full(original_shape, -np.inf, dtype=dtype)

        for window, (r, c) in zip(windows, coords):
            h_win, w_win = window.shape[:2]

            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Update the area with the element-wise maximum
            current_area = accumulator[r:r_end, c:c_end, :]
            new_data = window[:h_paste, :w_paste, :]

            accumulator[r:r_end, c:c_end, :] = np.maximum(current_area, new_data)

        # Replace remaining -inf with 0 (areas where no window was placed)
        accumulator[np.isinf(accumulator)] = 0.0

        return accumulator

    else:
        raise ValueError(f"Unknown mode: {mode}")


# --- Example Usage ---
if __name__ == "__main__":
    # # Dummy setup
    # orig_H, orig_W, C = 100, 100, 3
    # win_size = 20

    # # Window 1: value 10
    # w1 = np.ones((win_size, win_size, C)) * 10
    # # Window 2: value 50 (Overlaps w1)
    # w2 = np.ones((win_size, win_size, C)) * 50

    # windows_list = [w1, w2]
    # coords_list = [(0,0), (10,10)] # Overlapping at 10,10

    # # Test Average
    # res_avg = stitch_windows(windows_list, coords_list, (orig_H, orig_W, C), mode='average')
    # print(f"AVG Mode - Overlap Pixel (15,15): {res_avg[15,15,0]} (Expected 30.0)")

    # # Test Max
    # res_max = stitch_windows(windows_list, coords_list, (orig_H, orig_W, C), mode='max')
    # print(f"MAX Mode - Overlap Pixel (15,15): {res_max[15,15,0]} (Expected 50.0)")

    # # Test Empty
    # print(f"MAX Mode - Empty Pixel (80,80):   {res_max[80,80,0]} (Expected 0.0)")

    # TODO add mask for non-paded pixels
    from data_preparation.grid_loader.utils import load_raster

    df = pd.read_csv(
        "../yan_bp3/data_samples_approach_2/test_indices.csv"
    )  # pd.read_csv("../yan_bp3/data_samples_approach_1/test_indices.csv")
    # df = df[(df["season"]==1) &(df["cause"]==1)]
    base_dir = "../yan_bp3/data_samples_approach_2"
    original_hexel = load_raster(
        "../yan_bp3/hex41/outputs/hex_41_season_1_cause_1_bc.tif"
    ).data  # hex_41_season_1_cause_1_bc.tif hex_41_20000_iter_bp.tif
    H, W = original_hexel.shape[:2]
    original_hexel = original_hexel[:, :, np.newaxis]
    original_shape = original_hexel.shape
    all_data_points = []
    all_locations = []
    for data in np.array(df):
        path = data[0]
        array = np.load(os.path.join(base_dir, path))[:, :, -1]
        row, col = data[5], data[6]
        all_locations.append((row, col))
        all_data_points.append(array[:, :, np.newaxis])

    reconstructed_hexel = stitch_windows(all_data_points, all_locations, original_shape, mode="average")
    h_win, w_win = 128, 128
    win_diff = []
    for r, c in all_locations:
        r_end = min(r + h_win, original_shape[0])
        c_end = min(c + w_win, original_shape[1])
        win_diff.append(abs(np.sum(original_hexel[r:r_end, c:c_end, :] - reconstructed_hexel[r:r_end, c:c_end, :])))
    print("The difference between the 2 hexels is", np.sum(original_hexel.reshape(H, W) - reconstructed_hexel.reshape(H, W)))
    # print(win_diff)
    print(np.sum(win_diff))
    # print(original_hexel-reconstructed_hexel)
