import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

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

        data = np.load(file_path)
        
        # Ensure data is explicitly float32 (common source of PyTorch errors)
        data = data.astype(np.float32)
        input_arr, output_arr = data[:,:,:-2], data[:,:,-1]

        #TODO: remove NaNs from the inp data (replace by mean)

        return input_arr, output_arr