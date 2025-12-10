import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# Global Burn Count Min Max
BURN_COUNT_MAX = 1336.0
BURN_COUNT_MIN = 0.0


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


class GridDataset(Dataset):
    """
    Dataset class for loading the data
    """

    def __init__(self, csv_path: str, root_dir: str, filename_col: str = "filename", out_norm: str = "min_max", transform=None):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            out_norm (str): How to normalize the output burn counts. [Options: total_iters, season_cause_iters, min_max]
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata_df = pd.read_csv(csv_path)
        self.all_files = list(self.metadata_df[filename_col])
        self.root_dir = root_dir
        self.filename_col = filename_col
        self.transform = transform
        self.out_norm = out_norm
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

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        # 1. Get the filename from the CSV
        filename = self.all_files[idx]

        # 2. Construct full path
        file_path = os.path.join(self.root_dir, filename)

        data = np.load(file_path).astype(np.float32)
        input_arr, output_arr = data[:, :, :-1], data[:, :, -1]
        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = np.isnan(input_arr[:, :, 0])  # return a mask for the loss function
        input_arr = fill_nan_channel_mean_numpy(input_arr)  # remove NaNs from the inp data (replace by mean)

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
    transform: Callable | None = None,
):
    """
    Creates and returns a DataLoader
    """
    train_dataset = GridDataset(
        csv_path=train_csv_path, root_dir=root_dir, filename_col=filename_col, out_norm=out_norm, transform=transform
    )
    train_dataset = GridDataset(csv_path=train_csv_path, root_dir=root_dir, filename_col=filename_col, transform=transform)

    val_dataset = GridDataset(csv_path=val_csv_path, root_dir=root_dir, filename_col=filename_col, out_norm=out_norm, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_dataset = GridDataset(csv_path=val_csv_path, root_dir=root_dir, filename_col=filename_col, transform=transform)

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
    transform: Callable | None = None,
):
    """
    Creates and returns the test loader
    """
    test_dataset = GridDataset(csv_path=test_csv_path, root_dir=root_dir, filename_col=filename_col, out_norm=out_norm, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_dataset = GridDataset(csv_path=test_csv_path, root_dir=root_dir, filename_col=filename_col, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return test_loader


if __name__ == "__main__":
    out_norm = "min_max"
    train_loader, val_loader = get_train_val_dataloader(
        train_csv_path="../yan_bp3/data_samples_approach_2/train_indices.csv",
        val_csv_path="../yan_bp3/data_samples_approach_2/val_indices.csv",
        root_dir="../yan_bp3",
        out_norm=out_norm,
        batch_size=4,
        transform=None,
    )

    test_loader = get_test_loader(
        test_csv_path="../yan_bp3/data_samples_approach_2/test_indices.csv",
        root_dir="../yan_bp3",
        batch_size=4,
        out_norm=out_norm,
        transform=None,
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
