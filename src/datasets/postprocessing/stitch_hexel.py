import numpy as np
import pandas as pd


def stitch_windows(
    windows: list[np.ndarray],
    coords: list[tuple],
    masks: list[np.ndarray],
    original_shape: tuple,
    mode: str = "mean",
    window_size: int = 128,
    center_crop_size: int = 64,
) -> np.ndarray:
    """
    Reconstructs an image from overlapping windows using either averaging or maximization.

    Args:
        windows (list of np.array): List of window arrays (H_win, W_win, C).
        coords (list of tuples): List of (row, col) top-left coordinates for each window.
        masks (list of np.ndarray): List of masks for the windows. True if valid value
        original_shape (tuple): Shape of the target hexel (H, W, C).
        mode (str): How to combine/stitch the windows (Options: mean, max, center_crop)
    Returns:
        np.array: The reconstructed image (Shape: original_shape, (H,W))
    """
    dtype = np.float64

    if mode == "mean":
        # --- AVERAGE MODE ---
        accumulator = np.zeros(original_shape, dtype=dtype)
        counter = np.zeros(original_shape, dtype=dtype)

        for window, mask, (r, c) in zip(windows, masks, coords):
            window[~mask] = 0.0
            h_win, w_win = window.shape[:2]

            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Accumulate Sum and Count
            accumulator[r:r_end, c:c_end] += window[:h_paste, :w_paste]
            counter[r:r_end, c:c_end] += mask[:h_paste, :w_paste]

        # Normalize
        valid_mask = counter > 0
        reconstructed = np.zeros_like(accumulator)
        reconstructed[valid_mask] = accumulator[valid_mask] / counter[valid_mask]

        return reconstructed

    elif mode == "max":
        # --- MAX MODE ---
        # Initialize with negative infinity so any real data (even negative) will override it
        accumulator = np.full(original_shape, -np.inf, dtype=dtype)

        for window, mask, (r, c) in zip(windows, masks, coords):
            h_win, w_win = window.shape[:2]
            window[~mask] = -np.inf
            # Safe slicing
            r_end = min(r + h_win, original_shape[0])
            c_end = min(c + w_win, original_shape[1])
            h_paste = r_end - r
            w_paste = c_end - c

            # Update the area with the element-wise maximum
            current_area = accumulator[r:r_end, c:c_end]
            new_data = window[:h_paste, :w_paste]

            accumulator[r:r_end, c:c_end] = np.maximum(current_area, new_data)

        # Replace remaining -inf with 0 (areas where no window was placed)
        accumulator[np.isinf(accumulator)] = 0.0

        return accumulator

    elif mode == "center_crop":
        print("==========Stitch model is center crop===============")
        H, W = original_shape[:2]
        accumulator = np.zeros(original_shape, dtype=dtype)
        halo = (window_size - center_crop_size) // 2
        for window, mask, (r, c) in zip(windows, masks, coords):
            window[~mask] = 0.0
            center_pred = window[halo : halo + center_crop_size, halo : halo + center_crop_size]
            valid_h = min(center_crop_size, H - r)
            valid_w = min(center_crop_size, W - c)
            if valid_h > 0 and valid_w > 0:
                accumulator[r : r + valid_h, c : c + valid_w] = center_pred[:valid_h, :valid_w]
        return accumulator

    else:
        raise ValueError(f"Unknown mode: {mode}")
