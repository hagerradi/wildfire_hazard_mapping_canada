import numpy as np

from data_preparation.grid_loader.utils import fuel_ranking

AVAILABLE_DATA_SOURCES = ["grid", "weather"]
MAX_FUEL_GRID = float(max(fuel_ranking.values()))


def get_data_source_class(name: str):
    """
    Returns the source class dynamically to avoid circular imports.
    """
    if name == "grid":
        from src.datasets.sources import GridSource

        return GridSource
    elif name == "weather":
        from src.datasets.sources import WeatherSource

        return WeatherSource
    else:
        raise ValueError(f"Unknown data source type: {name}. Available: {AVAILABLE_DATA_SOURCES}")


def fill_nan_channel_mean_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Fills NaNs in a (H, W, C) array with the mean of the corresponding channel.
    Modifies the array in-place.
    """
    # 1. Calculate the mean of each channel, ignoring NaNs
    # axis=(0, 1) aggregates over Height and Width, leaving (C,)
    channel_means = np.nanmean(arr, axis=(0, 1))

    # 2. Find the indices where values are NaN
    # This returns a boolean mask of shape (H, W, C)
    nan_mask = np.isnan(arr)

    # 3. Replace NaNs
    # We grab the specific channel index (2) from the nan locations
    # and map them to the calculated means.
    arr[nan_mask] = np.take(channel_means, np.where(nan_mask)[2])

    return arr


def one_hot_encode(arr: np.ndarray, channel_idx: int, num_classes: int) -> np.ndarray:
    """
    Replaces the nth channel with its one-hot encoded version.
    Input: (H, W, C)
    Output: (H, W, C - 1 + num_classes)
    """
    # 1. Split the array
    # left: (H, W, n)
    left_part = arr[:, :, :channel_idx]

    # right: (H, W, C - n - 1)
    right_part = arr[:, :, channel_idx + 1 :]

    # target: (H, W)
    target_channel = arr[:, :, channel_idx]

    nan_mask = np.isnan(target_channel)

    # Replace NaN with 0 (or any safe index) temporarily so .astype(int) doesn't crash
    # We use np.nan_to_num to swap NaN -> 0 safely
    safe_target = np.nan_to_num(target_channel, nan=0).astype(int)

    # 3. One-Hot Encode using the "safe" integers
    encoded_part = np.eye(num_classes, dtype=arr.dtype)[safe_target]

    # 4. Zero out the vectors where the original value was NaN
    # Before this, the NaNs were encoded as Class 0 (because we filled with 0)
    # This step corrects that by setting them to [0, 0, 0...]
    encoded_part[nan_mask] = 0  # Nan is no fuel

    # 3. Concatenate along the channel axis (last axis)
    return np.concatenate([left_part, encoded_part.astype(np.float32), right_part], axis=-1)  # (H,W,C+14)


def log_norm(out_arr: np.ndarray, multiplier: int = 1000) -> np.ndarray:
    """
    Normalize the output burn prob array using log norm
    """
    return np.log1p(multiplier * out_arr) / np.log1p(multiplier)


def output_burn_prob_norm(output_arr: np.ndarray, burn_prob_max: float, burn_prob_min: float, out_norm: str) -> np.ndarray:
    """
    Normalize the output burn prob map
    """
    if out_norm == "min_max":
        output_arr = (output_arr - burn_prob_min) / (burn_prob_max - burn_prob_min)
        output_arr = np.clip(output_arr, 0.0, 1.0)
    elif out_norm == "log":
        output_arr = log_norm(output_arr).astype(np.float32)
    return output_arr
