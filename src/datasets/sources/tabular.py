import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import TabularParams
from src.datasets.sources.base import DataSource
from src.datasets.utils import FIRE_SIZE_MEANS


class TabularSource(DataSource):
    """
    Retrieves samples by mapping a zone ID from a spatial grid patch
    to a lookup table of tabular data (loaded from CSV).
    Samples are sampled according to sampling_approach.
    """

    def __init__(
        self,
        root_dir: str,
        params: TabularParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the .npy files.
            params (TabularParams): a TabularParams config. object.
            modelling_approach (str): The approach used for modelling.
        """
        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        self.csv_name = params.csv_name
        self.feature_names_list = params.feature_names_list
        self.fire_weather_zone_id_col = params.fire_weather_zone_id_col
        self.sampling_approach = params.sampling_approach
        self.num_samples_per_patch = params.num_samples_per_patch

        self.df = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        # 1. Extract weather zone channel index
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
            self.zone_channel = channel_feature_map["weather_grid"][0]
        # 2. Create weather lookup table for faster sampling
        self.lut = {}
        for zone, group in self.df.groupby(self.fire_weather_zone_id_col):
            feats = group[self.feature_names_list].values.astype(np.float32)
            self.lut[int(zone)] = feats

    def get_sample(self, patch_info: dict):
        if "data" in patch_info:
            # Fast load (Training)
            data = patch_info["data"]
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(patch_info["file_path"])
        zone_arr = data[:, :, self.zone_channel]
        mask = ~np.isnan(zone_arr)
        zone_arr = zone_arr[mask]
        sample_features = None
        values, counts = np.unique(zone_arr, return_counts=True)

        # 1. Select candidates depending on sampling approach
        if self.sampling_approach == "mode":
            mode_zone = int(values[np.argmax(counts)])
            candidates = self.lut.get(mode_zone)
        else:
            raise ValueError("Please provide a correct sampling approach. Valid approaches: ['mode'].")

        # 2. Perform actual sampling
        if candidates is None:  # Only happens in fire size distribution csv
            value = FIRE_SIZE_MEANS.get(
                self.feature_names_list[0]
            )  # TODO: Using mean imputation for now, will change once confirmed with experts
            sample_features = np.full(shape=(self.num_samples_per_patch, len(self.feature_names_list)), fill_value=value, dtype=np.float32)
        else:
            replace = len(candidates) < self.num_samples_per_patch
            sample_indices = np.random.choice(len(candidates), size=self.num_samples_per_patch, replace=replace)
            sample_features = candidates[sample_indices]

        return sample_features

    def input_dim(self):
        """Returns number of features for each item"""
        return len(self.feature_names_list)
