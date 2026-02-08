import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.datasets.sources.base import DataSource


class WeatherSource(DataSource):
    """
    Retrieves weather samples by mapping the fire weather zone in a grid
    patch to a lookup table of historical weather data (loaded from CSV).
    Weather samples are sampled according to sampling_approach.
    """

    def __init__(
        self,
        csv_name: str,
        root_dir: str,
        feature_names_list: list[str],
        sampling_approach: str = "mode",
        num_samples_per_patch: int = 128,
        modelling_approach: str = "1",
        transform: Callable | None = None,
    ):
        """
        Args:
            csv_name (str): Name of the historical weather CSV file.
            root_dir (str): Directory with all the .npy files.
            feature_names_list (list): List containing features we wish to include for model training.
            sampling_approach (str): Name of the sampling approach to select weather samples. Options = ['mode'].
            num_samples_per_patch (int): Number of weather samples to get for each patch grid.
            modelling_approach (str): The approach used for modelling.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv_name = csv_name
        self.root_dir = root_dir
        self.feature_names_list = feature_names_list
        self.sampling_approach = sampling_approach
        self.num_samples_per_patch = num_samples_per_patch
        self.modelling_approach = modelling_approach
        self.df_weather = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        # 1. Extract weather zone channel index
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
            self.weather_channel = channel_feature_map["weather_grid"][0]
        # 2. Create weather lookup table for faster sampling
        self.weather_lut = {}
        for zone, group in self.df_weather.groupby("wx_zone"):
            feats = group[self.feature_names_list].values.astype(np.float32)
            self.weather_lut[int(zone)] = feats

    def get_sample(self, context: dict):
        if "data" in context:
            # Fast load (Training)
            data = context["data"]
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(context["file_path"])
        weather_arr = data[:, :, self.weather_channel]
        mask = ~np.isnan(weather_arr)
        weather_arr = weather_arr[mask]
        weather_samples = None

        if self.sampling_approach == "mode":
            values, counts = np.unique(weather_arr, return_counts=True)
            mode_zone = int(values[np.argmax(counts)])
            candidates = self.weather_lut[mode_zone]
            if len(candidates) >= self.num_samples_per_patch:
                sample_indices = np.random.choice(len(candidates), size=self.num_samples_per_patch, replace=False)
                weather_samples = candidates[sample_indices]
            else:
                sample_indices = np.random.choice(len(candidates), size=self.num_samples_per_patch, replace=True)
                weather_samples = candidates[sample_indices]
        else:
            raise ValueError("Please provide a correct sampling approach. Valid approaches: ['mode'].")

        return weather_samples

    def input_dim(self):
        """Returns number of features for each weather item"""
        return len(self.feature_names_list)
