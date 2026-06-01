import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch

from src.config import SpatializedTabularParams
from src.datasets.sources.base import DataSource


class SpatializedTabularSource(DataSource):
    """Rasterize zone-level tabular features into patch-aligned image channels."""

    def __init__(
        self,
        root_dir: str,
        params: SpatializedTabularParams,
        modelling_approach: str = "1",
        transform: Callable | None = None,
    ):
        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform

        self.csv_name = params.csv_name
        self.feature_names_list = params.feature_names_list
        self.zone_id_col = params.fire_weather_zone_id_col
        self.zone_channel_key = params.zone_channel_key
        self.aggregation = params.aggregation.lower()
        self.include_missing_mask = params.include_missing_mask
        self.missing_value_strategy = params.missing_value_strategy.lower()
        self.shuffle_lut = params.shuffle_lut
        self.shuffle_seed = params.shuffle_seed

        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
        if self.zone_channel_key not in channel_feature_map:
            raise ValueError(
                f"Missing zone channel {self.zone_channel_key!r} in feature channel map. Available keys: {list(channel_feature_map)}"
            )
        self.zone_channel = channel_feature_map[self.zone_channel_key][0]

        df = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        missing_columns = [col for col in [self.zone_id_col, *self.feature_names_list] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in {self.csv_name!r}: {missing_columns}")

        self.lut = self._build_lut(df)
        if not self.lut:
            raise ValueError(
                f"Spatialized tabular LUT is empty for {self.csv_name!r}. Check zone column {self.zone_id_col!r} and selected features."
            )

        self.global_fill = df[self.feature_names_list].mean(axis=0).to_numpy(dtype=np.float32)
        self._validate_missing_value_strategy()
        if self.shuffle_lut:
            self._shuffle_lut_values()

    def _integer_zone_ids_from_csv(self, zone_ids: pd.Series) -> pd.Series:
        numeric_zone_ids = pd.to_numeric(zone_ids, errors="coerce")
        invalid_numeric_mask = zone_ids.notna() & numeric_zone_ids.isna()
        if invalid_numeric_mask.any():
            bad_value = zone_ids[invalid_numeric_mask].iloc[0]
            raise ValueError(f"Zone column {self.zone_id_col!r} in {self.csv_name!r} contains non-numeric zone id {bad_value!r}.")

        present_mask = numeric_zone_ids.notna()
        present_values = numeric_zone_ids[present_mask].to_numpy(dtype=np.float64)
        non_finite_mask = ~np.isfinite(present_values)
        if non_finite_mask.any():
            bad_value = float(present_values[non_finite_mask][0])
            raise ValueError(f"Zone column {self.zone_id_col!r} in {self.csv_name!r} contains non-finite zone id {bad_value}.")

        rounded_values = np.rint(present_values)
        if not np.allclose(present_values, rounded_values, atol=1e-3):
            bad_value = float(present_values[np.argmax(np.abs(present_values - rounded_values))])
            raise ValueError(f"Zone column {self.zone_id_col!r} in {self.csv_name!r} contains non-integer zone id {bad_value}.")

        integer_zone_ids = pd.Series(pd.NA, index=zone_ids.index, dtype="Int64")
        integer_zone_ids.loc[present_mask] = rounded_values.astype(np.int64)
        return integer_zone_ids

    def _build_lut(self, df: pd.DataFrame) -> dict[int, np.ndarray]:
        aggregations = {
            "mean": "mean",
            "median": "median",
            "min": "min",
            "max": "max",
        }
        if self.aggregation not in aggregations:
            raise ValueError(f"Unsupported spatialized tabular aggregation {self.aggregation!r}. Supported values: {sorted(aggregations)}")

        normalized_df = df.copy()
        zone_id_key = "__spatialized_tabular_zone_id"
        normalized_df[zone_id_key] = self._integer_zone_ids_from_csv(df[self.zone_id_col])
        grouped = normalized_df.groupby(zone_id_key, dropna=True)[self.feature_names_list].agg(aggregations[self.aggregation])
        grouped = grouped.dropna(how="any")
        return {int(zone): row.to_numpy(dtype=np.float32) for zone, row in grouped.iterrows()}

    def _validate_missing_value_strategy(self) -> None:
        if self.missing_value_strategy not in {"global_mean", "zero", "raise"}:
            raise ValueError(
                f"Unsupported missing_value_strategy={self.missing_value_strategy!r}. Supported values: ['global_mean', 'zero', 'raise']."
            )

    def _shuffle_lut_values(self) -> None:
        zones = sorted(self.lut)
        values = [self.lut[zone].copy() for zone in zones]
        rng = np.random.default_rng(self.shuffle_seed)
        permutation = rng.permutation(len(values))
        self.lut = {zone: values[permutation[index]] for index, zone in enumerate(zones)}

    def _initial_features(self, height: int, width: int) -> np.ndarray:
        if self.missing_value_strategy == "global_mean":
            fill_value = self.global_fill
        elif self.missing_value_strategy == "zero":
            fill_value = np.zeros(len(self.feature_names_list), dtype=np.float32)
        else:
            fill_value = np.full(len(self.feature_names_list), np.nan, dtype=np.float32)
        return np.broadcast_to(fill_value, (height, width, len(self.feature_names_list))).copy()

    def _zone_ids(self, zone_grid: np.ndarray, finite_zone_mask: np.ndarray) -> np.ndarray:
        zone_int_grid = np.zeros(zone_grid.shape, dtype=np.int64)
        if not finite_zone_mask.any():
            return zone_int_grid

        finite_zone_values = zone_grid[finite_zone_mask]
        rounded_zone_values = np.rint(finite_zone_values)
        if not np.allclose(finite_zone_values, rounded_zone_values, atol=1e-3):
            bad_value = float(finite_zone_values[np.argmax(np.abs(finite_zone_values - rounded_zone_values))])
            raise ValueError(f"Zone channel {self.zone_channel_key!r} contains non-integer zone id {bad_value}.")

        zone_int_grid[finite_zone_mask] = rounded_zone_values.astype(np.int64)
        return zone_int_grid

    def get_sample(self, patch_info: dict):
        data = patch_info["data"] if "data" in patch_info else np.load(patch_info["file_path"], mmap_mode="r")
        zone_grid = np.asarray(data[:, :, self.zone_channel])
        height, width = zone_grid.shape
        features = self._initial_features(height, width)

        finite_zone_mask = np.isfinite(zone_grid) & (zone_grid > 0)
        zone_int_grid = self._zone_ids(zone_grid=zone_grid, finite_zone_mask=finite_zone_mask)
        matched_mask = np.zeros((height, width), dtype=bool)
        for zone in np.unique(zone_int_grid[finite_zone_mask]):
            zone_features = self.lut.get(int(zone))
            if zone_features is None:
                continue
            pixel_mask = finite_zone_mask & (zone_int_grid == zone)
            features[pixel_mask] = zone_features
            matched_mask[pixel_mask] = True

        missing_mask = ~matched_mask
        if self.missing_value_strategy == "raise" and missing_mask.any():
            missing_zones = sorted({int(z) for z in zone_int_grid[finite_zone_mask & missing_mask]})
            raise ValueError(f"Spatialized tabular source {self.csv_name!r} has missing LUT zones: {missing_zones[:20]}")

        if self.include_missing_mask:
            features = np.concatenate([features, missing_mask[:, :, None].astype(np.float32)], axis=-1)

        return torch.from_numpy(features.astype(np.float32)).permute(2, 0, 1)

    def input_dim(self):
        return len(self.feature_names_list) + int(self.include_missing_mask)
