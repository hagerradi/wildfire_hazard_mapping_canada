import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Global Burn Count Min Max
BURN_COUNT_MAX = 1336.0
BURN_COUNT_MIN = 0.0

MAX_FUEL_GRID = 20.0
MIN_FUEL_GRID = 0.0


def fill_nan_channel_mean_numpy(arr):
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


def one_hot_encode(arr, channel_idx, num_classes):
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

    # target: (H, W) - We cast to int for indexing
    target_channel = arr[:, :, channel_idx].astype(int)

    nan_mask = np.isnan(target_channel)

    # Replace NaN with 0 (or any safe index) temporarily so .astype(int) doesn't crash
    # We use np.nan_to_num to swap NaN -> 0 safely
    safe_target = np.nan_to_num(target_channel, nan=0).astype(int)

    # 3. One-Hot Encode using the "safe" integers
    encoded_part = np.eye(num_classes, dtype=arr.dtype)[safe_target]

    # 4. Zero out the vectors where the original value was NaN
    # Before this, the NaNs were encoded as Class 0 (because we filled with 0)
    # This step corrects that by setting them to [0, 0, 0...]
    encoded_part[nan_mask] = 0

    # 3. Concatenate along the channel axis (last axis)
    return np.concatenate([left_part, encoded_part.astype(float), right_part], axis=-1)


class GridDataset(Dataset):
    """
    Dataset class for loading the data
    """

    def __init__(
        self,
        csv_path: str,
        root_dir: str,
        filename_col: str = "filename",
        out_norm: str = "min_max",
        fuel_feats_processing: str = "ordinal",
        fuel_feats_ordinal_is_norm: bool | None = True,
        modelling_approach: str = "2",
        transform: Callable | None = None,
        feature_names_list: list[str] | None = None,
    ):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            out_norm (str): How to normalize the output burn counts. [Options: total_iters, season_cause_iters, min_max]
            modelling_approach (str): The approach used for modelling
            transform (callable, optional): Optional transform to be applied on a sample.
            feature_names_list (list): List of features being used for training ((options: None or feature list) All feats: ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid"])
        """
        self.metadata_df = pd.read_csv(csv_path)
        self.all_files = list(self.metadata_df[filename_col])
        self.root_dir = root_dir
        self.filename_col = filename_col
        self.transform = transform
        self.out_norm = out_norm
        self.fuel_feats_processing = fuel_feats_processing
        self.fuel_feats_ordinal_is_norm = fuel_feats_ordinal_is_norm
        if self.out_norm == "total_iters":
            self.out_norm_array = list(
                self.metadata_df["total_unique_iters"]
            )  # total number of unique interations that produced fires for all seasons and causes
        elif self.out_norm == "season_cause_iters":
            self.out_norm_array = list(
                self.metadata_df["season_cause_unique_iters"]
            )  # total number of unique interations that produced fires for a single season and cause
        else:
            self.out_norm_array = [1] * len(self.all_files)  # if we want to predict the counts

        self.channel_indices = None
        with open(os.path.join(root_dir, f"feature_channel_maps/feature_channel_map_{modelling_approach}.json"), "r") as f:
            channel_feature_map = json.load(f)
        self.fuel_feat_index = channel_feature_map["fuel_grid"][0]
        if feature_names_list:
            self.channel_indices = [item for key in feature_names_list for item in channel_feature_map[key]]
            print(self.channel_indices)
            self.fuel_feat_index = self.channel_indices.index(self.fuel_feat_index)

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # 1. Get the filename from the CSV
        filename = self.all_files[idx]

        # 2. Construct full path
        file_path = os.path.join(self.root_dir, filename)

        data = np.load(file_path).astype(np.float32)
        input_arr, output_arr = data[:, :, :-1], data[:, :, -1]
        input_arr = input_arr[:, :, self.channel_indices] if self.channel_indices else input_arr

        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = np.isnan(input_arr[:, :, 0])  # return a mask for the loss function
        # Processing one hot encoding
        if self.fuel_feats_processing == "one_hot":
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_index, num_classes=int(MAX_FUEL_GRID + 1))

        input_arr = fill_nan_channel_mean_numpy(input_arr)  # remove NaNs from the inp data (replace by mean)

        # Processing ordinal encoding norm (if not norm do nothing)
        if self.fuel_feats_processing == "ordinal":
            input_arr[:, :, self.fuel_feat_index][mask] = 0.0
            if self.fuel_feats_ordinal_is_norm:
                input_arr[:, :, self.fuel_feat_index] = (input_arr[:, :, self.fuel_feat_index] - MIN_FUEL_GRID) / (
                    MAX_FUEL_GRID - MIN_FUEL_GRID
                )

        # TODO/Assumption: this min_max normalization supports season/cause scenarios only
        if self.out_norm == "min_max":
            output_arr = (output_arr - BURN_COUNT_MIN) / (BURN_COUNT_MAX - BURN_COUNT_MIN)
        else:
            output_arr /= self.out_norm_array[idx]
        return (
            torch.from_numpy(input_arr),
            torch.from_numpy(np.expand_dims(output_arr, -1)),
            torch.from_numpy(mask.astype(np.uint8)),
        )  # (H,W,C), (H,W,1), (H,W)


def get_train_val_dataloader(
    train_csv_path: str,
    val_csv_path: str,
    root_dir: str,
    filename_col: str = "filename",
    batch_size: int = 4,
    num_workers: int = 0,
    out_norm: str = "min_max",
    fuel_feats_processing: str = "ordinal",
    fuel_feats_ordinal_is_norm: bool | None = True,
    modelling_approach: str = "2",
    transform: Callable | None = None,
    feature_names_list: list[str] | None = None,
):
    """
    Creates and returns a DataLoader
    """
    train_dataset = GridDataset(
        csv_path=train_csv_path,
        root_dir=root_dir,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_processing=fuel_feats_processing,
        fuel_feats_ordinal_is_norm=fuel_feats_ordinal_is_norm,
        modelling_approach=modelling_approach,
        transform=transform,
        feature_names_list=feature_names_list,
    )
    val_dataset = GridDataset(
        csv_path=val_csv_path,
        root_dir=root_dir,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_processing=fuel_feats_processing,
        fuel_feats_ordinal_is_norm=fuel_feats_ordinal_is_norm,
        modelling_approach=modelling_approach,
        transform=transform,
        feature_names_list=feature_names_list,
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader


def get_test_loader(
    test_csv_path: str,
    root_dir: str,
    filename_col: str = "filename",
    batch_size: int = 4,
    num_workers: int = 0,
    out_norm: str = "min_max",
    fuel_feats_processing: str = "ordinal",
    fuel_feats_ordinal_is_norm: bool | None = True,
    modelling_approach: str = "2",
    transform: Callable | None = None,
    feature_names_list: list[str] | None = None,
):
    """
    Creates and returns the test loader
    """
    test_dataset = GridDataset(
        csv_path=test_csv_path,
        root_dir=root_dir,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_processing=fuel_feats_processing,
        fuel_feats_ordinal_is_norm=fuel_feats_ordinal_is_norm,
        modelling_approach=modelling_approach,
        transform=transform,
        feature_names_list=feature_names_list,
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader


if __name__ == "__main__":
    out_norm = "min_max"
    # feature_names_list = ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid"]
    feature_names_list = ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid"]
    modelling_approach = "2"
    train_loader, val_loader = get_train_val_dataloader(
        train_csv_path="../yan_bp3/data_samples_approach_2/train_indices.csv",
        val_csv_path="../yan_bp3/data_samples_approach_2/val_indices.csv",
        root_dir="../yan_bp3",
        out_norm=out_norm,
        fuel_feats_processing="one_hot",
        fuel_feats_ordinal_is_norm=True,
        modelling_approach=modelling_approach,
        batch_size=4,
        transform=None,
        feature_names_list=feature_names_list,
    )

    test_loader = get_test_loader(
        test_csv_path="../yan_bp3/data_samples_approach_2/test_indices.csv",
        root_dir="../yan_bp3",
        batch_size=4,
        out_norm=out_norm,
        fuel_feats_processing="one_hot",
        fuel_feats_ordinal_is_norm=True,
        modelling_approach=modelling_approach,
        transform=None,
        feature_names_list=feature_names_list,
    )

    print("\nIterating through Train DataLoader:")
    for batch_idx, (data, target, mask) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {target.shape}, mask: {mask.shape}")
        print("NAN values in the loaded data", torch.isnan(data).sum().item())
        break
    print("\nIterating through val DataLoader:")
    for batch_idx, (data, target, _) in enumerate(val_loader):
        print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {target.shape}")
        print("NAN values in the loaded data", torch.isnan(data).sum().item())
        break

    print("\nIterating through test DataLoader:")
    for batch_idx, (data, target, _) in enumerate(test_loader):
        print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {target.shape}")
        print("NAN values in the loaded data", torch.isnan(data).sum().item())
        break
