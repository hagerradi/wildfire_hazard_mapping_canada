import json
import os

import numpy as np

from data_preparation.grid_loader.utils import fuel_ranking

MAX_FUEL_GRID = float(max(fuel_ranking.values()))


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


def output_burn_prob_norm(
    output_arr: np.ndarray, burn_prob_max: float, burn_prob_min: float, out_norm: str
) -> np.ndarray:
    """
    Normalize the output burn prob map
    """
    if out_norm == "min_max":
        output_arr = (output_arr - burn_prob_min) / (burn_prob_max - burn_prob_min)
        output_arr = np.clip(output_arr, 0.0, 1.0)
    elif out_norm == "log":
        output_arr = log_norm(output_arr).astype(np.float32)
    return output_arr


def compute_number_input_channels(
    feature_names_list: list[str], fuel_feats_encoding: str, root_dir: str, modelling_approach: str, output_mult: bool
) -> int:
    """
    Calculates the total number of input channels based on selected features
    and encoding strategy.
    """
    if not feature_names_list:  # Catch None and []
        raise ValueError("feature_names_list cannot be None or empty")

    # 1. Load the feature map
    feature_channel_map_path = os.path.join(root_dir, f"feature_channel_map_{modelling_approach}.json")

    if not os.path.exists(feature_channel_map_path):
        raise FileNotFoundError(f"Feature map not found at: {feature_channel_map_path}")

    with open(feature_channel_map_path) as f:
        channel_feature_map = json.load(f)

    # 2. Calculate base channels from the feature map
    # This sums up the channels for every feature in your list
    total_channels = 0
    for name in feature_names_list:
        if name in channel_feature_map:
            total_channels += len(channel_feature_map[name])
        else:
            raise ValueError(f"Feature '{name}' not found in feature_channel_map.")

    # 3. Adjust for One-Hot Encoding of Fuel
    # If using one-hot, we remove the original 'fuel_grid' channel (1)
    # and add the one-hot vectors (MAX_FUEL_GRID + 1)
    if "fuel_grid" in feature_names_list and fuel_feats_encoding == "one_hot":
        num_fuel_classes = int(MAX_FUEL_GRID + 1)
        # Net change: -1 (remove ordinal) + num_classes (add one-hot)
        total_channels = total_channels - 1 + num_fuel_classes
    
    if output_mult:
        total_channels += 1

    return total_channels
