from abc import ABC, abstractmethod
from collections.abc import Callable
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
import json
import numpy as np
from data_preparation.grid_loader.utils import BURN_COUNT_MAX, BURN_COUNT_MIN, fuel_ranking, get_range_burn_prob
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_burn_prob_norm
MAX_FUEL_GRID = float(max(fuel_ranking.values()))
MIN_FUEL_GRID = float(min(fuel_ranking.values()))

class MultiSourceDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        indices: DataIndex,
        sources: dict[str, DataSource]
        ):
        self.indices = indices
        self.sources = sources

    def __getitem__(self, idx):
        context = self.indices.get_context(idx)
        context['data'] = np.load(context['file_path'], mmap_mode="r") # Load once, distribute where needed
        sample = {}
        for name, source in self.sources.items():
            sample[name] = source.get_sample(context)
        return sample

    def __len__(self):
        return len(self.indices)


#%%
indices = DataIndex(
    csv_name = "train_indices.csv",
    root_dir="/home/mila/o/oseaj/scratch/nrcan_data/data_samples_approach_1_weather"
    )

grid_source = GridSource(
    root_dir="/home/mila/o/oseaj/scratch/nrcan_data/data_samples_approach_1_weather",
    feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"]
    )

weather_source = WeatherSource(
    csv_name="weather_table.csv", 
    root_dir="/home/mila/o/oseaj/scratch/nrcan_data/data_samples_approach_1_weather",
    features_names_list=["temp", "rh", "prec", "ffmc", "dmc", "dc", "isi", "bui"],
    )


#%%
ds = MultiSourceDataset(
    indices=indices,
    sources={
        "weather": weather_source,
        "grid": grid_source
    }
)

#%%
ds = MultiSourceDataset(
    indices=indices,
    sources={
        "weather": weather_source,
    }
)
# %%
dl = DataLoader(ds, batch_size=5)
batch = next(iter(dl))
# %%
# %%
weather_arr = batch['weather']
# %%

# %%
input_arr, output_arr, mask = batch['grid']
# %%
input_arr.shape, output_arr.shape, weather_arr.shape
