import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from config import DataConfig
from data_preparation.grid_loader.utils import fuel_ranking, get_range_burn_prob

# Global Burn Count Min Max
BURN_COUNT_MAX = 1336.0
BURN_COUNT_MIN = 0.0

MAX_FUEL_GRID = float(max(fuel_ranking.values()))
MIN_FUEL_GRID = float(min(fuel_ranking.values()))


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


class GridDataset(Dataset):
    """
    Dataset class for loading the data
    """

    def __init__(
        self,
        csv_name: str,
        root_dir: str,
        feature_names_list: list[str],
        filename_col: str = "filename",
        out_norm: str = "min_max",
        fuel_feats_encoding: str = "ordinal",
        normalize_fuel_feats_ordinal: bool | None = True,
        modelling_approach: str = "2",
        valid_mask_threshold: float = 0.01,
        transform: Callable | None = None,
    ):
        """
        Args:
            csv_name (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            out_norm (str): How to normalize the output burn counts for modelling approach 2. [Options: total_iters, season_cause_iters, min_max]
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            mask_threshold (float): The threshold for how much valid data should be present in a data sample
            transform (callable, optional): Optional transform to be applied on a sample.
            feature_names_list (list): List of features being used for training ((options: None or feature list) All feats: ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid"])
        """
        self.filename_col = filename_col
        self.transform = transform
        self.out_norm = out_norm
        self.modelling_approach = modelling_approach
        self.root_dir = root_dir
        self.valid_mask_threshold = valid_mask_threshold
        self.feature_names_list = feature_names_list

        if not self.feature_names_list:
            raise ValueError(
                "Feature names list should never be empty or None. Valid list: [ignition_grid, esc_fires_grid, fuel_grid, elevation_grid, weather_grid, wind_grid]"
            )

        self.metadata_df = pd.read_csv(os.path.join(self.root_dir, csv_name))
        self.metadata_df = self.metadata_df[self.metadata_df["valid_ratio"] > self.valid_mask_threshold]
        self.all_files = list(self.metadata_df[filename_col])

        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.BURN_PROB_MAX, self.BURN_PROB_MIN = get_range_burn_prob(os.path.dirname(self.root_dir))

        self.fuel_feats_encoding = fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = normalize_fuel_feats_ordinal
        if self.out_norm == "total_iters":
            self.out_norm_array = list(
                self.metadata_df["total_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for all seasons and causes
        elif self.out_norm == "season_cause_iters":
            self.out_norm_array = list(
                self.metadata_df["season_cause_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for a single season and cause

        self.channel_indices = None

        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
            self.channel_indices = [item for key in self.feature_names_list for item in channel_feature_map[key]]
            if "fuel_grid" in self.feature_names_list:
                self.fuel_feat_index = channel_feature_map["fuel_grid"][0]

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # 1. Get the filename from the CSV
        filename = self.all_files[idx]

        # 2. Construct full path
        file_path = os.path.join(self.root_dir, filename)

        data = np.load(file_path).astype(np.float32)
        input_arr, output_arr = data[:, :, :-1], data[:, :, -1]

        output_arr[np.isnan(output_arr)] = 0.0

        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = ~np.isnan(input_arr[:, :, 0])  # mask is True where not NaN, False where NaN

        # Processing one hot encoding
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":  # (H,W,C+20)
            num_classes = int(MAX_FUEL_GRID + 1)
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_index, num_classes=num_classes)
            # Update channel_indices to account for new one-hot channels
            old_fuel_idx = self.fuel_feat_index
            self.channel_indices = [i for i in self.channel_indices if i != old_fuel_idx]
            # Insert new indices for the one-hot channels at the position of the old fuel index
            self.channel_indices = (
                self.channel_indices[:old_fuel_idx]
                + list(range(old_fuel_idx, old_fuel_idx + num_classes))
                + [i + num_classes - 1 for i in self.channel_indices[old_fuel_idx:]]
            )

        input_arr = fill_nan_channel_mean_numpy(input_arr)  # remove NaNs from the inp data (replace by mean)

        # Processing ordinal encoding norm (if not norm do nothing)
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "ordinal":
            input_arr[:, :, self.fuel_feat_index][~mask] = 0.0  # Nan is no fuel
            if self.normalize_fuel_feats_ordinal:
                input_arr[:, :, self.fuel_feat_index] = (input_arr[:, :, self.fuel_feat_index] - MIN_FUEL_GRID) / (
                    MAX_FUEL_GRID - MIN_FUEL_GRID
                )

        input_arr = input_arr[:, :, self.channel_indices] if self.channel_indices else input_arr

        if self.modelling_approach == "2":
            if self.out_norm == "min_max":
                output_arr = (output_arr - BURN_COUNT_MIN) / (BURN_COUNT_MAX - BURN_COUNT_MIN)
            elif self.out_norm in ["total_iters", "season_cause_iters"]:
                output_arr /= self.out_norm_array[idx]
        elif self.modelling_approach == "1" and self.out_norm == "min_max":
            output_arr = (output_arr - self.BURN_PROB_MIN) / (self.BURN_PROB_MAX - self.BURN_PROB_MIN)
            output_arr = np.clip(output_arr, 0.0, 1.0)

        return (
            torch.from_numpy(input_arr).permute(2, 0, 1),
            torch.from_numpy(np.expand_dims(output_arr, 0)),
            torch.from_numpy(np.expand_dims(mask, 0)),  # keep as boolean for efficiency
        )  # (C, H, W), (1, H, W), (1, H, W)


def get_train_val_dataloader(
    config: DataConfig,
    modelling_approach: str = "1",
):
    """
    Creates and returns a DataLoader
    """
    root_dir = config.root_dir
    train_csv_name = config.train_split
    val_csv_name = config.val_split
    filename_col = config.filename_col
    batch_size = config.batch_size
    num_workers = config.num_workers
    out_norm = config.output_normalization
    transform = config.transform
    feature_names_list = config.feature_names_list
    fuel_feats_encoding = config.fuel_feats_encoding
    normalize_fuel_feats_ordinal = config.normalize_fuel_feats_ordinal
    valid_mask_threshold = config.valid_mask_threshold

    train_dataset = GridDataset(
        csv_name=train_csv_name,
        root_dir=root_dir,
        feature_names_list=feature_names_list,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_encoding=fuel_feats_encoding,
        normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
        modelling_approach=modelling_approach,
        valid_mask_threshold=valid_mask_threshold,
        transform=transform,
    )
    val_dataset = GridDataset(
        csv_name=val_csv_name,
        root_dir=root_dir,
        feature_names_list=feature_names_list,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_encoding=fuel_feats_encoding,
        normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
        modelling_approach=modelling_approach,
        valid_mask_threshold=valid_mask_threshold,
        transform=transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_test_loader(
    config: DataConfig,
    modelling_approach: str = "1",
):
    """
    Creates and returns the test loader
    """
    root_dir = config.root_dir
    test_csv_name = config.test_split
    filename_col = config.filename_col
    batch_size = config.batch_size
    num_workers = config.num_workers
    out_norm = config.output_normalization
    transform = config.transform
    feature_names_list = config.feature_names_list
    fuel_feats_encoding = config.fuel_feats_encoding
    normalize_fuel_feats_ordinal = config.normalize_fuel_feats_ordinal
    valid_mask_threshold = config.valid_mask_threshold

    test_dataset = GridDataset(
        csv_name=test_csv_name,
        root_dir=root_dir,
        feature_names_list=feature_names_list,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_encoding=fuel_feats_encoding,
        normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
        modelling_approach=modelling_approach,
        valid_mask_threshold=valid_mask_threshold,
        transform=transform,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader
