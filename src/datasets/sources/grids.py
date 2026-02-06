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

class GridSource(DataSource):
    def __init__(
        self, 
        root_dir: str,
        feature_names_list: list[str],
        out_norm: str = "min_max",
        fuel_feats_encoding: str = "ordinal",
        normalize_fuel_feats_ordinal: bool | None = True,
        modelling_approach: str = "1",
        transform: Callable | None = None
        ):
        self.root_dir = root_dir
        self.out_norm = out_norm
        self.feature_names_list = feature_names_list
        self.fuel_feats_encoding = fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = normalize_fuel_feats_ordinal
        self.modelling_approach = modelling_approach
        self.transform = transform
        
        # 1. Normalizations
        self.BURN_PROB_MAX, self.BURN_PROB_MIN = 1.0, 0.0
        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.BURN_PROB_MAX, self.BURN_PROB_MIN = get_range_burn_prob(os.path.dirname(self.root_dir))
        if self.out_norm == "total_iters":
            self.out_norm_array = list(
                self.metadata_df["total_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for all seasons and causes
        elif self.out_norm == "season_cause_iters":
            self.out_norm_array = list(
                self.metadata_df["season_cause_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for a single season and cause

        # 2. Update indices
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            self.channel_feature_map = json.load(f)
            self.input_channel_indices = [item for key in self.feature_names_list for item in self.channel_feature_map[key]]
            print(self.input_channel_indices)
            if "fuel_grid" in self.feature_names_list:
                self.fuel_feat_index = self.channel_feature_map["fuel_grid"][0]
                if self.fuel_feats_encoding == "one_hot":
                    num_classes = int(MAX_FUEL_GRID + 1)
                    # Update input_channel_indices to account for new one-hot channels
                    idx_fuel_feats = self.input_channel_indices.index(self.fuel_feat_index)
                    self.input_channel_indices = self.input_channel_indices[:idx_fuel_feats] + self.input_channel_indices[idx_fuel_feats + 1 :]
                    # Insert new indices for the one-hot channels at the position of the old fuel index
                    self.input_channel_indices = (
                        self.input_channel_indices[:idx_fuel_feats]
                        + list(range(self.fuel_feat_index, self.fuel_feat_index + num_classes))
                        + [i + num_classes - 1 for i in self.input_channel_indices[idx_fuel_feats:]]
                    )

    def get_sample(self, context: dict):
        if 'data' in context:
            # Fast Path (Training)
            data = context['data'].astype(np.float32)
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(context['file_path']).astype(np.float32)

        # 1. Separate inputs, output, and mask
        input_arr, output_arr = data[:, :, :-1], data[:, :, -1]
        output_arr[np.isnan(output_arr)] = 0.0
        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = ~np.isnan(input_arr[:, :, 0])  # mask is True where not NaN, False where NaN
        
        # 2. Processing one hot encoding
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":  # (H,W,C+20)
            num_classes = int(MAX_FUEL_GRID + 1)
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_index, num_classes=num_classes)
        
        # 3. Mean Imputation
        input_arr = fill_nan_channel_mean_numpy(input_arr) 

        # 4. Process ordinal encoding norm (if not norm do nothing)
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "ordinal":
            input_arr[:, :, self.fuel_feat_index][~mask] = 0.0  # Nan is no fuel
            if self.normalize_fuel_feats_ordinal:
                input_arr[:, :, self.fuel_feat_index] = (input_arr[:, :, self.fuel_feat_index] - MIN_FUEL_GRID) / (
                    MAX_FUEL_GRID - MIN_FUEL_GRID
                )

        # 5. Filter to just chosen input channel indices or if no features selected just return None
        input_arr = input_arr[:, :, self.input_channel_indices] if self.input_channel_indices else None

        # 6. Perform normalizations
        if self.modelling_approach == "2":
            if self.out_norm == "min_max":
                output_arr = (output_arr - BURN_COUNT_MIN) / (BURN_COUNT_MAX - BURN_COUNT_MIN)
            elif self.out_norm in ["total_iters", "season_cause_iters"]:
                output_arr /= self.out_norm_array[idx]
        elif self.modelling_approach == "1":
            output_arr = output_burn_prob_norm(
                output_arr=output_arr, burn_prob_max=self.BURN_PROB_MAX, burn_prob_min=self.BURN_PROB_MIN, out_norm=self.out_norm
            )

        input_arr = torch.from_numpy(input_arr).permute(2, 0, 1) if input_arr else None
        output_arr = torch.from_numpy(np.expand_dims(output_arr, 0))
        mask = torch.from_numpy(np.expand_dims(mask, 0))  # keep as boolean for efficiency

        # 7. Apply transforms if provided
        if self.transform:
            input_arr, output_arr, mask = self.transform(input_arr, output_arr, mask)

        return (input_arr, output_arr, mask)  # (C, H, W), (1, H, W), (1, H, W)


    def input_dim(self):
        """TODO implement"""
        return None

