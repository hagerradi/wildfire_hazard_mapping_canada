import json
import os
from collections.abc import Callable

import numpy as np
import torch

from data_preparation.spatial.utils import FUEL_GROUP_MAP, get_output_log_stats, get_range_elevation, get_range_output
from src.config import GridParams
from src.datasets.sources.base import DataSource
from src.datasets.targets import get_target_spec
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_target_norm


class GridSource(DataSource):
    """
    DataSource class for Spatial Grid
    """

    def __init__(
        self,
        root_dir: str,
        params: GridParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
        raw_data_dir: str | None = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the .npy files.
            feature_names_list (list): List of features being used for training ((options: None or feature list)
                All feats: ["ignition_grid", "fuel_grid", "elevation_grid", "wind_grid"])
            target_name (str): Output target to train against. Supported: bp, fi, ros.
            out_norm (str): How to normalize the output target.
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            transform (callable, optional): Optional transform to be applied on a sample.
            raw_data_dir (str, optional): Directory with raw per-hexel rasters used for normalization ranges.
        """

        self.root_dir = root_dir
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else os.path.dirname(self.root_dir)
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        self.target = get_target_spec(params.target_name)
        self.feature_names_list = params.feature_names_list
        self.out_norm = params.out_norm
        self.target_log_mean = params.target_log_mean
        self.target_log_std = params.target_log_std
        self.fuel_feats_encoding = params.fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = params.normalize_fuel_feats_ordinal
        self.num_fuel_classes = int(max(FUEL_GROUP_MAP.values()) + 1)

        # 1. Normalizations (for modelling approach 1)
        self.TARGET_MAX, self.TARGET_MIN = 1.0, 0.0
        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.TARGET_MAX, self.TARGET_MIN = get_range_output(self.raw_data_dir, self.target.output_type)
        elif self.modelling_approach == "1" and self.out_norm == "log_standard":
            if self.target_log_mean is None or self.target_log_std is None:
                self.target_log_mean, self.target_log_std = get_output_log_stats(self.raw_data_dir, self.target.output_type)
            if self.target_log_std <= 0.0:
                raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {self.target_log_std}.")

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
            output_channel_indices = self.channel_feature_map.get(self.target.channel_key)
            if not output_channel_indices:
                raise ValueError(
                    f"Missing output channel in feature channel map. Expected {self.target.channel_key!r} "
                    f"for target_name={self.target.name!r}, "
                    f"found keys: {list(self.channel_feature_map.keys())}"
                )
            self.output_channel_index = output_channel_indices[0]
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
        self.ELEVATION_MAX, self.ELEVATION_MIN = get_range_elevation(self.raw_data_dir)

    def get_sample(self, patch_info: dict):
        data = patch_info["data"].astype(np.float32) if "data" in patch_info else np.load(patch_info["file_path"]).astype(np.float32)

        # 1. Separate inputs, output, and mask
        input_arr, output_arr = data[:, :, self.preprocess_channel_indices], data[:, :, self.output_channel_index]
        input_arr_for_mask = input_arr[:, :, self.raw_input_local_indices]
        assert np.all(np.isnan(input_arr_for_mask) == np.isnan(input_arr_for_mask[..., :1])), "NaN mask differs across channels!"
        target_mask = ~np.isnan(output_arr)
        output_arr = np.where(target_mask, output_arr, 0.0)
        mask = ~np.isnan(input_arr_for_mask[:, :, 0]) & target_mask  # True where both inputs and target are valid.

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
            output_arr = output_target_norm(
                output_arr=output_arr,
                target_max=self.TARGET_MAX,
                target_min=self.TARGET_MIN,
                out_norm=self.out_norm,
                target_log_mean=self.target_log_mean,
                target_log_std=self.target_log_std,
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
