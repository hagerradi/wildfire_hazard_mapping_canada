import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


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
    def __init__(self, csv_path:str, root_dir:str, filename_col:str='filename', transform=None):
        """
        Args:
            csv_path (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            label_col (str, optional): Column name in CSV containing labels.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_frame = pd.read_csv(csv_path)
        self.all_files = list(self.data_frame[filename_col])
        self.root_dir = root_dir
        self.filename_col = filename_col
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 1. Get the filename from the CSV
        filename = self.all_files[idx]
        
        # 2. Construct full path
        file_path = os.path.join(self.root_dir, filename)

        data = np.load(file_path).astype(np.float32)
        input_arr, output_arr = data[:,:,:-2], data[:,:,-1]
        mask = np.isnan(input_arr[:,:,0]) #return a mask for the loss function
        input_arr = fill_nan_channel_mean_numpy(input_arr) #remove NaNs from the inp data (replace by mean)
        #TODO: process the output counts
        return torch.from_numpy(input_arr), torch.from_numpy(output_arr), torch.from_numpy(mask)
    
def get_train_val_dataloader(train_csv_path, val_csv_path, root_dir, batch_size=4, shuffle=True, num_workers=0, transform=None):
    """
    Creates and returns a DataLoader 
    """
    train_dataset = GridDataset(
        csv_path=train_csv_path, 
        root_dir=root_dir, 
        filename_col='filename', 
        transform=transform
    )

    val_dataset = GridDataset(
        csv_path=val_csv_path, 
        root_dir=root_dir, 
        filename_col='filename', 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader

if __name__ == "__main__":
    train_loader, val_loader = get_train_val_dataloader(
        train_csv_path="../yan_bp3/data_samples_approach_2/train.csv",
        val_csv_path="../yan_bp3/data_samples_approach_2/val.csv",
        root_dir="./",
        batch_size=4,
        transform=None
    )

    print("\nIterating through Train DataLoader:")
    for batch_idx, (data, target, mask) in enumerate(train_loader):
        print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {target.shape}, mask: {mask.shape}")
        print("NAN values in the loaded data",torch.isnan(data).sum().item())
        break
    print("\nIterating through val DataLoader:")
    for batch_idx, (data, target,_) in enumerate(val_loader):
        print(f"Batch {batch_idx}: Data Shape: {data.shape}, Labels: {target}")
        print("NAN values in the loaded data",torch.isnan(data).sum().item())
        break