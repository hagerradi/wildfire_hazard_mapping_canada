import json
import math
import os
from collections.abc import Callable

import numpy as np
import torch

from data_preparation.paths import Paths, normalize_mask_scope
from data_preparation.spatial.utils import (
    FUEL_GROUP_MAP,
    get_output_log_stats_cached,
    get_range_elevation,
    get_range_output,
    load_spatial_raster,
    read_split_hex_ids,
)
from src.config import GridParams
from src.datasets.sources.base import DataSource
from src.datasets.targets import get_target_specs
from src.datasets.utils import (
    apply_bp_nodata_zero_range,
    default_target_norm,
    fill_nan_channel_mean_numpy,
    one_hot_encode,
    output_burn_prob_norm,
)


class GridSource(DataSource):
    """
    DataSource class for Spatial Grid
    """

    _ALLOWED_TERRAIN_DERIVATIVES = {"slope", "aspect_sin", "aspect_cos"}
    # Input validity is determined solely by the input channels' own finiteness; target coverage never
    # restricts which pixels count as valid model inputs. "input_only" is the only supported policy.
    _ALLOWED_INPUT_MASK_POLICIES = {"input_only"}

    def __init__(
        self,
        root_dir: str,
        params: GridParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
        raw_data_dir: str | None = None,
        train_split_csv_name: str | None = None,
    ):
        """
        Args:
            root_dir (str): Directory with all the .npy files.
            feature_names_list (list): List of features being used for training ((options: None or feature list)
                All feats: ["ignition_grid", "fuel_grid", "elevation_grid", "wind_grid"])
            target_name (str): Output target to train against. Supported: bp, fi, ros.
            out_norm (str): How to normalize the output burn counts for modelling approach 2. [Options: total_iters, season_cause_iters, min_max]
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            transform (callable, optional): Optional transform to be applied on a sample.
            raw_data_dir (str, optional): Directory with raw per-hexel rasters used for normalization ranges.
        """

        self.root_dir = root_dir
        self.raw_data_dir = raw_data_dir if raw_data_dir is not None else self.root_dir
        self._validate_raw_ranges = raw_data_dir is not None
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        # Normalization statistics must be derived from the training split only, so held-out
        # hexes never leak into target/elevation normalization constants. None preserves the
        # legacy full-scan behaviour (e.g. single-hex inference where no split is provided).
        self._train_hex_ids: set[int] | None = None
        if train_split_csv_name is not None:
            self._train_hex_ids = read_split_hex_ids(os.path.join(self.root_dir, train_split_csv_name))

        self.targets = get_target_specs(params.target_name)
        self.feature_names_list = params.feature_names_list
        self.out_norm = params.out_norm
        self.target_out_norms = params.target_out_norms
        self.target_log_mean = params.target_log_mean
        self.target_log_std = params.target_log_std
        self.target_log_means = params.target_log_means
        self.target_log_stds = params.target_log_stds
        self.fuel_feats_encoding = params.fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = params.normalize_fuel_feats_ordinal
        self.include_hex_coords = params.include_hex_coords
        self.terrain_derivatives = params.terrain_derivatives
        self.terrain_cell_size_m = params.terrain_cell_size_m
        self.bp_nodata_as_zero = params.bp_nodata_as_zero
        self.input_mask_policy = params.input_mask_policy
        self.num_fuel_classes = int(max(FUEL_GROUP_MAP.values()) + 1)
        self._hex_shape_cache: dict[tuple[str, str], tuple[int, int]] = {}
        self.elevation_input_channel_index: int | None = None

        if self.input_mask_policy not in self._ALLOWED_INPUT_MASK_POLICIES:
            raise ValueError(
                f"Invalid input_mask_policy={self.input_mask_policy!r}. "
                f"Allowed options are: {sorted(self._ALLOWED_INPUT_MASK_POLICIES)}"
            )

        unknown_terrain_derivatives = set(self.terrain_derivatives) - self._ALLOWED_TERRAIN_DERIVATIVES
        if unknown_terrain_derivatives:
            raise ValueError(
                f"Invalid terrain_derivatives {sorted(unknown_terrain_derivatives)}. "
                f"Allowed options are: {sorted(self._ALLOWED_TERRAIN_DERIVATIVES)}"
            )
        if self.terrain_derivatives and "elevation_grid" not in self.feature_names_list:
            raise ValueError("terrain_derivatives requires 'elevation_grid' in feature_names_list.")

        # 1. Normalizations (for modelling approach 1)
        self.target_ranges = {target.name: (1.0, 0.0) for target in self.targets}
        if self.modelling_approach == "1":
            for target in self.targets:
                if self._target_out_norm(target.name) != "min_max":
                    continue
                try:
                    target_max, target_min = get_range_output(self.raw_data_dir, target.output_type, self._train_hex_ids)
                except ValueError:
                    if self._validate_raw_ranges:
                        raise
                    target_max, target_min = 1.0, 0.0
                target_max, target_min = apply_bp_nodata_zero_range(
                    target_name=target.name,
                    max_value=target_max,
                    min_value=target_min,
                    bp_nodata_as_zero=self.bp_nodata_as_zero,
                )
                self.target_ranges[target.name] = (target_max, target_min)
                if self._validate_raw_ranges:
                    self._validate_range(
                        max_value=target_max,
                        min_value=target_min,
                        label=target.label,
                        source_dir=self.raw_data_dir,
                    )

            for target in self.targets:
                if self._target_out_norm(target.name) != "log_standard":
                    continue
                mean = self.target_log_means.get(target.name, self.target_log_mean)
                std = self.target_log_stds.get(target.name, self.target_log_std)
                if mean is None or std is None:
                    mean, std = get_output_log_stats_cached(
                        self.root_dir,
                        target.output_type,
                        allowed_hex_ids=self._train_hex_ids,
                        raw_data_dir=self.raw_data_dir,
                    )
                    if target.name in self.target_log_stds or target.name in self.target_log_means:
                        self.target_log_means[target.name] = mean
                        self.target_log_stds[target.name] = std
                    else:
                        self.target_log_mean = mean
                        self.target_log_std = std

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
            self.output_channel_indices = []
            for target in self.targets:
                output_channel_indices = self.channel_feature_map.get(target.channel_key)
                if not output_channel_indices:
                    raise ValueError(
                        f"Missing output channel in feature channel map. Expected {target.channel_key!r} "
                        f"for target_name={target.name!r}, "
                        f"found keys: {list(self.channel_feature_map.keys())}"
                    )
                self.output_channel_indices.append(output_channel_indices[0])
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
            if "elevation_grid" in self.feature_names_list:
                elev_feat_local_index = self.channel_index_to_local_index[self.channel_feature_map["elevation_grid"][0]]
                elev_feat_encoded_index = elev_feat_local_index
                if (
                    "fuel_grid" in self.feature_names_list
                    and self.fuel_feats_encoding == "one_hot"
                    and self.fuel_feat_local_index < elev_feat_local_index
                ):
                    elev_feat_encoded_index += self.num_fuel_classes - 1
                self.elevation_input_channel_index = self.input_channel_indices.index(elev_feat_encoded_index)
        # 3. normalization for elevation grid
        try:
            self.ELEVATION_MAX, self.ELEVATION_MIN = get_range_elevation(self.raw_data_dir, self._train_hex_ids)
        except ValueError:
            if self._validate_raw_ranges:
                raise
            self.ELEVATION_MAX, self.ELEVATION_MIN = 1.0, 0.0
        if self._validate_raw_ranges:
            self._validate_range(
                max_value=self.ELEVATION_MAX,
                min_value=self.ELEVATION_MIN,
                label="elevation",
                source_dir=self.raw_data_dir,
            )

    @staticmethod
    def _validate_range(max_value: float, min_value: float, label: str, source_dir: str) -> None:
        if not np.isfinite(max_value) or not np.isfinite(min_value) or max_value <= min_value:
            raise ValueError(
                f"Invalid {label} normalization range from raw_data_dir={source_dir!r}: "
                f"min={min_value}, max={max_value}. Check that raw_data_dir points to the raw hexel dataset, "
                "not only the prepared patch directory."
            )

    def _target_out_norm(self, target_name: str) -> str:
        if target_name in self.target_out_norms:
            return self.target_out_norms[target_name]
        if len(self.targets) > 1:
            return default_target_norm(target_name)
        return self.out_norm

    def _target_log_stats(self, target_name: str) -> tuple[float | None, float | None]:
        mean = self.target_log_means.get(target_name, self.target_log_mean)
        std = self.target_log_stds.get(target_name, self.target_log_std)
        return mean, std

    @staticmethod
    def _hex_id_to_key(hex_id: object) -> str:
        return str(int(float(str(hex_id)))).zfill(2)

    def _get_full_hex_shape(self, patch_info: dict, patch_height: int, patch_width: int) -> tuple[int, int]:
        if "full_height" in patch_info and "full_width" in patch_info:
            return int(patch_info["full_height"]), int(patch_info["full_width"])

        if "hex_id" not in patch_info:
            raise KeyError("include_hex_coords=True requires patch metadata with 'hex_id'.")

        hex_id = self._hex_id_to_key(patch_info["hex_id"])
        mask_scope = normalize_mask_scope(str(patch_info.get("mask_scope", "actual")))
        cache_key = (hex_id, mask_scope)
        if cache_key in self._hex_shape_cache:
            return self._hex_shape_cache[cache_key]

        paths = Paths(hex_id=hex_id, root_dir=self.raw_data_dir)
        elevation_grid, _ = load_spatial_raster(
            path=paths.elevation_grid(hex_id=hex_id),
            mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
        )
        full_shape = tuple(int(dim) for dim in elevation_grid.shape)
        if len(full_shape) != 2 or full_shape[0] < patch_height or full_shape[1] < patch_width:
            raise ValueError(
                f"Invalid full hex shape {full_shape} for hex {hex_id}; expected at least patch shape {(patch_height, patch_width)}."
            )
        self._hex_shape_cache[cache_key] = full_shape
        return full_shape

    def _append_hex_coord_channels(self, input_arr: np.ndarray, patch_info: dict) -> np.ndarray:
        missing = [key for key in ("row", "col") if key not in patch_info]
        if missing:
            raise KeyError(f"include_hex_coords=True requires patch metadata columns: {missing}.")

        patch_height, patch_width = input_arr.shape[:2]
        full_height, full_width = self._get_full_hex_shape(patch_info, patch_height, patch_width)
        row = int(patch_info["row"])
        col = int(patch_info["col"])

        denom_y = max(full_height - 1, 1)
        denom_x = max(full_width - 1, 1)
        y_coords = 2.0 * (row + np.arange(patch_height, dtype=np.float32)) / denom_y - 1.0
        x_coords = 2.0 * (col + np.arange(patch_width, dtype=np.float32)) / denom_x - 1.0
        y_grid = np.clip(y_coords[:, None], -1.0, 1.0)
        x_grid = np.clip(x_coords[None, :], -1.0, 1.0)
        coord_arr = np.stack(
            [
                np.broadcast_to(y_grid, (patch_height, patch_width)),
                np.broadcast_to(x_grid, (patch_height, patch_width)),
            ],
            axis=-1,
        ).astype(np.float32)
        return np.concatenate([input_arr, coord_arr], axis=-1)

    @staticmethod
    def _finite_difference(values: torch.Tensor, dim: int, spacing: float) -> torch.Tensor:
        grad = torch.zeros_like(values)
        size = values.shape[dim]
        if size < 2:
            return grad

        if dim == 0:
            grad[0, :] = (values[1, :] - values[0, :]) / spacing
            grad[-1, :] = (values[-1, :] - values[-2, :]) / spacing
            if size > 2:
                grad[1:-1, :] = (values[2:, :] - values[:-2, :]) / (2.0 * spacing)
        elif dim == 1:
            grad[:, 0] = (values[:, 1] - values[:, 0]) / spacing
            grad[:, -1] = (values[:, -1] - values[:, -2]) / spacing
            if size > 2:
                grad[:, 1:-1] = (values[:, 2:] - values[:, :-2]) / (2.0 * spacing)
        else:
            raise ValueError(f"Expected dim 0 or 1 for finite differences, got {dim}.")
        return grad

    def _terrain_derivative_channels(self, input_arr: torch.Tensor) -> list[torch.Tensor]:
        if self.elevation_input_channel_index is None:
            raise RuntimeError("terrain_derivatives requires a resolved elevation input channel.")

        elevation_norm = input_arr[self.elevation_input_channel_index]
        elevation_m = elevation_norm * (self.ELEVATION_MAX - self.ELEVATION_MIN) + self.ELEVATION_MIN
        dz_drow = self._finite_difference(elevation_m, dim=0, spacing=self.terrain_cell_size_m)
        dz_dcol = self._finite_difference(elevation_m, dim=1, spacing=self.terrain_cell_size_m)
        gradient_magnitude = torch.sqrt(dz_drow.square() + dz_dcol.square())
        flat_mask = gradient_magnitude <= 1e-12

        # Aspect is encoded as the downslope unit vector in raster coordinates.
        channel_map = {}
        if "slope" in self.terrain_derivatives:
            channel_map["slope"] = torch.atan(gradient_magnitude) / (math.pi / 2.0)
        if "aspect_sin" in self.terrain_derivatives:
            aspect_sin = -dz_dcol / gradient_magnitude.clamp_min(1e-12)
            channel_map["aspect_sin"] = torch.where(flat_mask, torch.zeros_like(aspect_sin), aspect_sin)
        if "aspect_cos" in self.terrain_derivatives:
            aspect_cos = dz_drow / gradient_magnitude.clamp_min(1e-12)
            channel_map["aspect_cos"] = torch.where(flat_mask, torch.zeros_like(aspect_cos), aspect_cos)

        return [channel_map[name].to(dtype=input_arr.dtype) for name in self.terrain_derivatives]

    def _append_terrain_derivative_channels(self, input_arr: torch.Tensor) -> torch.Tensor:
        if not self.terrain_derivatives:
            return input_arr
        terrain_channels = self._terrain_derivative_channels(input_arr)
        return torch.cat([input_arr, *[channel.unsqueeze(0) for channel in terrain_channels]], dim=0)

    def get_sample(self, patch_info: dict):
        data = patch_info["data"].astype(np.float32) if "data" in patch_info else np.load(patch_info["file_path"]).astype(np.float32)

        # 1. Separate inputs, outputs, and mask
        input_arr, output_arr = data[:, :, self.preprocess_channel_indices], data[:, :, self.output_channel_indices]
        input_arr_for_mask = input_arr[:, :, self.raw_input_local_indices]
        assert np.all(np.isnan(input_arr_for_mask) == np.isnan(input_arr_for_mask[..., :1])), "NaN mask differs across channels!"
        raw_target_mask = np.isfinite(output_arr)
        target_mask = raw_target_mask.copy()

        if self.bp_nodata_as_zero:
            for channel_idx, target in enumerate(self.targets):
                if target.name == "bp":
                    target_mask[:, :, channel_idx] = True
        output_arr = np.where(raw_target_mask, output_arr, 0.0)
        input_mask = np.isfinite(input_arr_for_mask[:, :, 0])
        mask = input_mask[:, :, np.newaxis] & target_mask  # True where both inputs and each target are valid.

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
        if self.include_hex_coords:
            input_arr = self._append_hex_coord_channels(input_arr, patch_info)

        # 7. Perform output normalizations
        if self.modelling_approach == "1":
            normalized_outputs = []
            for channel_idx, target in enumerate(self.targets):
                target_max, target_min = self.target_ranges[target.name]
                target_log_mean, target_log_std = self._target_log_stats(target.name)
                normalized_outputs.append(
                    output_burn_prob_norm(
                        output_arr=output_arr[:, :, channel_idx],
                        burn_prob_max=target_max,
                        burn_prob_min=target_min,
                        out_norm=self._target_out_norm(target.name),
                        target_log_mean=target_log_mean,
                        target_log_std=target_log_std,
                    )
                )
            output_arr = np.stack(normalized_outputs, axis=-1)

        if input_arr is not None:
            input_arr = torch.from_numpy(input_arr).permute(2, 0, 1)
        output_arr = torch.from_numpy(output_arr).permute(2, 0, 1)
        mask = torch.from_numpy(mask).permute(2, 0, 1)  # keep as boolean for efficiency

        # 7. Apply transforms if provided
        if self.transform:
            input_arr, output_arr, mask = self.transform(input_arr, output_arr, mask)
        input_arr = self._append_terrain_derivative_channels(input_arr)

        return (input_arr, output_arr, mask)  # (C, H, W), (num_targets, H, W), (num_targets, H, W)

    def input_dim(self):
        """
        Returns the number of input channels (C)
        Returns 0 if no fatures are selected (e.g. when only looking at target)
        """
        terrain_dim = len(self.terrain_derivatives)
        if self.input_channel_indices:
            return len(self.input_channel_indices) + (2 if self.include_hex_coords else 0) + terrain_dim
        if self.include_hex_coords:
            return 2 + terrain_dim
        return 0
