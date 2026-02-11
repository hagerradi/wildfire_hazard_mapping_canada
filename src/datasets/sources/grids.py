import json
import os
from collections.abc import Callable

import numpy as np
import torch

from data_preparation.grid_loader.utils import BURN_COUNT_MAX, BURN_COUNT_MIN, fuel_ranking, get_range_burn_prob
from src.datasets.sources.base import DataSource
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_burn_prob_norm


class GridSource(DataSource):
    """
    DataSource class for Spatial Grid
    """

    def __init__(
        self,
        root_dir: str,
        feature_names_list: list[str],
        out_norm: str = "min_max",
        fuel_feats_encoding: str = "one_hot",
        normalize_fuel_feats_ordinal: bool | None = True,
        modelling_approach: str = "1",
        transform: Callable | None = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the .npy files.
            feature_names_list (list): List of features being used for training ((options: None or feature list)
                All feats: ["ignition_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid"])
            out_norm (str): How to normalize the output burn counts for modelling approach 2. [Options: total_iters, season_cause_iters, min_max]
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.out_norm = out_norm
        self.feature_names_list = feature_names_list
        self.fuel_feats_encoding = fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = normalize_fuel_feats_ordinal
        self.modelling_approach = modelling_approach
        self.transform = transform
        self.norm_col_map = {"total_iters": "total_unique_iters", "season_cause_iters": "season_cause_unique_iters"}
        self.max_fuel_grid = float(max(fuel_ranking.values()))
        self.min_fuel_grid = float(min(fuel_ranking.values()))

        # 1. Normalizations (for modelling approach 1)
        self.BURN_PROB_MAX, self.BURN_PROB_MIN = 1.0, 0.0
        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.BURN_PROB_MAX, self.BURN_PROB_MIN = get_range_burn_prob(os.path.dirname(self.root_dir))

        # 2. Update indices
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            self.channel_feature_map = json.load(f)
            self.input_channel_indices = [item for key in self.feature_names_list for item in self.channel_feature_map[key]]
            if "fuel_grid" in self.feature_names_list:
                self.fuel_feat_index = self.channel_feature_map["fuel_grid"][0]
                if self.fuel_feats_encoding == "one_hot":
                    num_classes = int(self.max_fuel_grid + 1)
                    # Update input_channel_indices to account for new one-hot channels
                    idx_fuel_feats = self.input_channel_indices.index(self.fuel_feat_index)
                    self.input_channel_indices = (
                        self.input_channel_indices[:idx_fuel_feats] + self.input_channel_indices[idx_fuel_feats + 1 :]
                    )
                    # Insert new indices for the one-hot channels at the position of the old fuel index
                    self.input_channel_indices = (
                        self.input_channel_indices[:idx_fuel_feats]
                        + list(range(self.fuel_feat_index, self.fuel_feat_index + num_classes))
                        + [i + num_classes - 1 for i in self.input_channel_indices[idx_fuel_feats:]]
                    )

    def get_sample(self, patch_info: dict):
        if "data" in patch_info:
            # Fast Path (Training)
            data = patch_info["data"].astype(np.float32)
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(patch_info["file_path"]).astype(np.float32)

        # 1. Separate inputs, output, and mask
        input_arr, output_arr = data[:, :, :-1], data[:, :, -1]
        output_arr[np.isnan(output_arr)] = 0.0
        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = ~np.isnan(input_arr[:, :, 0])  # mask is True where not NaN, False where NaN

        # 2. Processing one hot encoding
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":  # (H,W,C+20)
            num_classes = int(self.max_fuel_grid + 1)
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_index, num_classes=num_classes)

        # 3. Mean Imputation
        input_arr = fill_nan_channel_mean_numpy(input_arr)

        # 4. Process ordinal encoding norm (if not norm do nothing)
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "ordinal":
            input_arr[:, :, self.fuel_feat_index][~mask] = 0.0  # Nan is no fuel
            if self.normalize_fuel_feats_ordinal:
                input_arr[:, :, self.fuel_feat_index] = (input_arr[:, :, self.fuel_feat_index] - self.min_fuel_grid) / (
                    self.max_fuel_grid - self.min_fuel_grid
                )

        # 5. Filter to just chosen input channel indices or if no features selected just return None
        if self.input_channel_indices is not None:
            input_arr = input_arr[:, :, self.input_channel_indices]

        # 6. Perform output normalizations
        if self.modelling_approach == "2":
            if self.out_norm == "min_max":
                output_arr = (output_arr - BURN_COUNT_MIN) / (BURN_COUNT_MAX - BURN_COUNT_MIN)
            elif self.out_norm in ["total_iters", "season_cause_iters"]:
                col_name = self.norm_col_map[self.out_norm]
                output_arr /= float(patch_info[col_name])
        elif self.modelling_approach == "1":
            output_arr = output_burn_prob_norm(
                output_arr=output_arr, burn_prob_max=self.BURN_PROB_MAX, burn_prob_min=self.BURN_PROB_MIN, out_norm=self.out_norm
            )

        if input_arr is not None:
            input_arr = torch.from_numpy(input_arr).permute(2, 0, 1)
        output_arr = torch.from_numpy(np.expand_dims(output_arr, 0))
        mask = torch.from_numpy(np.expand_dims(mask, 0))  # keep as boolean for efficiency

        # 7. Apply transforms if provided
        if self.transform:
            input_arr, output_arr, mask = self.transform(input_arr, output_arr, mask)

        return (input_arr, output_arr, mask)  # (C, H, W), (1, H, W), (1, H, W)

    def input_dim(self):
        """
        Returns the number of input channels (C)
        Returns 0 if no fatures are selected (e.g. when only looking at target)
        """
        if self.input_channel_indices:
            return len(self.input_channel_indices)
        return 0
