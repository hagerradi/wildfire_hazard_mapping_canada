import json
import os
from collections.abc import Callable

import numpy as np
import torch

from data_preparation.spatial.utils import FUEL_GROUP_MAP, get_range_elevation, get_range_output
from src.config import GridParams
from src.datasets.sources.base import DataSource
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_burn_prob_norm


class GridSource(DataSource):
    """
    DataSource class for Spatial Grid
    """

    OUTPUT_CHANNEL_KEYS = ("bp_out_grid", "esc_fires_grid")

    def __init__(
        self,
        root_dir: str,
        params: GridParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the .npy files.
            feature_names_list (list): List of features being used for training ((options: None or feature list)
                All feats: ["ignition_grid", "fuel_grid", "elevation_grid", "wind_grid"])
            out_norm (str): How to normalize the output burn counts for modelling approach 2. [Options: total_iters, season_cause_iters, min_max]
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        self.feature_names_list = params.feature_names_list
        self.out_norm = params.out_norm
        self.fuel_feats_encoding = params.fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = params.normalize_fuel_feats_ordinal
        self.num_fuel_classes = int(max(FUEL_GROUP_MAP.values()) + 1)

        self.norm_col_map = {"total_iters": "total_unique_iters", "season_cause_iters": "season_cause_unique_iters"}

        # 1. Normalizations (for modelling approach 1)
        self.BURN_PROB_MAX, self.BURN_PROB_MIN = 1.0, 0.0
        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.BURN_PROB_MAX, self.BURN_PROB_MIN = get_range_output(os.path.dirname(self.root_dir), "fire_burn_probability")

        # 2. Update indices
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            self.channel_feature_map = json.load(f)
            self.raw_input_channel_indices = [item for key in self.feature_names_list for item in self.channel_feature_map[key]]
            self.preprocess_channel_indices = sorted(set(self.raw_input_channel_indices))
            self.channel_index_to_local_index = {
                channel_index: local_index for local_index, channel_index in enumerate(self.preprocess_channel_indices)
            }
            self.raw_input_local_indices = [
                self.channel_index_to_local_index[channel_index] for channel_index in self.raw_input_channel_indices
            ]
            self.input_channel_indices = list(self.raw_input_local_indices)
            self.output_channel_index = next(
                (
                    channel_indices[0]
                    for channel_key in self.OUTPUT_CHANNEL_KEYS
                    for channel_indices in [self.channel_feature_map.get(channel_key)]
                    if channel_indices
                ),
                None,
            )
            if self.output_channel_index is None:
                raise ValueError(
                    f"Missing output channel in feature channel map. Expected one of {self.OUTPUT_CHANNEL_KEYS}, "
                    f"found keys: {list(self.channel_feature_map.keys())}"
                )
            if "fuel_grid" in self.feature_names_list:
                self.fuel_feat_index = self.channel_feature_map["fuel_grid"][0]
                self.fuel_feat_local_index = self.channel_index_to_local_index[self.fuel_feat_index]
                if self.fuel_feats_encoding == "one_hot":
                    updated_input_channel_indices = []
                    for channel_index in self.raw_input_local_indices:
                        if channel_index < self.fuel_feat_local_index:
                            updated_input_channel_indices.append(channel_index)
                        elif channel_index == self.fuel_feat_local_index:
                            updated_input_channel_indices.extend(
                                range(
                                    self.fuel_feat_local_index,
                                    self.fuel_feat_local_index + self.num_fuel_classes,
                                )
                            )
                        else:
                            updated_input_channel_indices.append(channel_index + self.num_fuel_classes - 1)
                    self.input_channel_indices = updated_input_channel_indices
        # 3. normalization for elevation grid
        self.ELEVATION_MAX, self.ELEVATION_MIN = get_range_elevation(os.path.dirname(self.root_dir))

    def get_sample(self, patch_info: dict):
        if "data" in patch_info:
            # Fast Path (Training)
            data = patch_info["data"].astype(np.float32)
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(patch_info["file_path"]).astype(np.float32)

        # 1. Separate inputs, output, and mask
        input_arr, output_arr = data[:, :, self.preprocess_channel_indices], data[:, :, self.output_channel_index]
        output_arr[np.isnan(output_arr)] = 0.0
        input_arr_for_mask = input_arr[:, :, self.raw_input_local_indices]
        assert np.all(np.isnan(input_arr_for_mask) == np.isnan(input_arr_for_mask[..., :1])), "NaN mask differs across channels!"
        mask = ~np.isnan(input_arr_for_mask[:, :, 0])  # mask is True where not NaN, False where NaN

        # 2. Normalize elevation (and any other input)
        if "elevation_grid" in self.feature_names_list:
            elev_feat_local_index = self.channel_index_to_local_index[self.channel_feature_map["elevation_grid"][0]]
            input_arr[:, :, elev_feat_local_index] = (input_arr[:, :, elev_feat_local_index] - self.ELEVATION_MIN) / (
                self.ELEVATION_MAX - self.ELEVATION_MIN + 1e-8
            )

        # 3. Processing one hot encoding
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":  # (H,W,C+20)
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_local_index, num_classes=self.num_fuel_classes)

        # 4. Mean Imputation
        input_arr = fill_nan_channel_mean_numpy(input_arr)

        # 6. Filter to just chosen input channel indices or if no features selected just return None
        if self.input_channel_indices is not None:
            input_arr = input_arr[:, :, self.input_channel_indices]

        # 7. Perform output normalizations
        if self.modelling_approach == "1":
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

        if len(self.feature_names_list) == 1 and "wind_grid" in self.feature_names_list:
            return input_arr
        return (input_arr, output_arr, mask)  # (C, H, W), (1, H, W), (1, H, W)

    def input_dim(self):
        """
        Returns the number of input channels (C)
        Returns 0 if no fatures are selected (e.g. when only looking at target)
        """
        if self.input_channel_indices:
            return len(self.input_channel_indices)
        return 0
