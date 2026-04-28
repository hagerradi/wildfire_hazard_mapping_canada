import json
import os
from collections.abc import Callable

import numpy as np
import pandas as pd

from src.config import TabularParams
from src.datasets.sources.base import DataSource


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
        self.fire_weather_zone_selection_approach = params.fire_weather_zone_selection_approach
        self.sampling_bias = params.sampling_bias
        self.feature_to_bias = params.feature_to_bias
        self.num_samples_per_patch = params.num_samples_per_patch

        # Pre-compute the column index for the bias feature
        self.bias_col_idx: int | None = None
        if self.sampling_bias is not None:
            if self.feature_to_bias not in self.feature_names_list:
                raise ValueError(
                    f"feature_to_bias '{self.feature_to_bias}' not found in feature_names_list. "
                    f"Valid features are: {self.feature_names_list}"
                )
            self.bias_col_idx = self.feature_names_list.index(self.feature_to_bias)

        self.df = pd.read_csv(os.path.join(self.root_dir, self.csv_name))
        # 1. Extract weather zone channel index
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            channel_feature_map = json.load(f)
            self.zone_channel = channel_feature_map["firezones_grid"][0]
        # 2. Create weather lookup table for faster sampling
        self.lut = {}
        for zone, group in self.df.groupby(self.fire_weather_zone_id_col):
            feats = group[self.feature_names_list].values.astype(np.float32)
            self.lut[int(zone)] = feats

        if not self.lut:
            raise ValueError(
                f"Weather LUT is empty: no valid zones found in '{self.csv_name}'. "
                f"Check that '{self.fire_weather_zone_id_col}' column contains valid zone IDs."
            )
        # Pre-built global fallback: used when a patch's zone IDs are absent from the LUT
        self._fallback_candidates = np.concatenate(list(self.lut.values()))

    def get_sample(self, patch_info: dict):
        if "data" in patch_info:
            # Fast load (Training)
            data = patch_info["data"]
        else:
            # Slow Path (Debugging / Standalone)
            data = np.load(patch_info["file_path"])
        zone_arr = data[:, :, self.zone_channel]
        mask = ~np.isnan(zone_arr) & (zone_arr > 0)
        zone_arr = zone_arr[mask]

        candidates = None
        weights = None
        values, counts = np.unique(zone_arr, return_counts=True)

        # 1. Select candidates depending on sampling approach
        if len(values) == 0:
            # Patch is fully masked — fall back to global candidates
            print("[TabularSource] Warning: patch is fully masked (all NaN). Using global fallback.")
            candidates = self._fallback_candidates
        elif self.fire_weather_zone_selection_approach == "mode":  # Selects the candidates from the most common zone in the patch
            # Try zones in descending frequency order until one is found in the LUT
            for zone_val in values[np.argsort(counts)[::-1]]:
                zone_cands = self.lut.get(int(zone_val))
                if zone_cands is not None:
                    candidates = zone_cands
                    break
            if candidates is None:
                print(
                    f"[TabularSource] Warning: no LUT match for any zone in patch (zones={[int(v) for v in values]}). Using global fallback."
                )
                candidates = self._fallback_candidates
        elif (
            self.fire_weather_zone_selection_approach == "weighted"
        ):  # Selects candidates from all zones in the patch, with probability proportional to their frequency
            all_candidates = []
            probs = []
            for val, count in zip(values, counts):
                zone_cands = self.lut.get(int(val))
                if zone_cands is not None:
                    all_candidates.append(zone_cands)
                    probs.append(np.full(len(zone_cands), count / len(zone_cands)))
            if not all_candidates:
                print(
                    f"[TabularSource] Warning: no LUT match for any zone in patch (zones={[int(v) for v in values]}). Using global fallback."
                )
                candidates = self._fallback_candidates
            else:
                candidates = np.concatenate(all_candidates)
                weights = np.concatenate(probs)
                weights /= weights.sum()
        else:
            raise ValueError(f"Unknown zone_selection_approach: {self.fire_weather_zone_selection_approach}")

        # 2. If sampling bias for a feature is specified, adjust weights accordingly
        if self.sampling_bias is not None and candidates is not None and len(candidates) > 0:
            bias_values = candidates[:, self.bias_col_idx]
            if self.sampling_bias == "high_values":  # Bias towards higher values of the feature
                bias_weights = np.clip(bias_values, 0.0, None)  # Clamp negatives to 0
            elif not self.sampling_bias:  # No bias, uniform sampling among candidates
                bias_weights = None
            else:
                raise ValueError(f"Unknown sampling_bias: {self.sampling_bias}")

            if bias_weights is not None:
                bias_sum = bias_weights.sum()
                if bias_sum > 0:
                    bias_weights /= bias_sum
                    # Compose with existing zone weights (if any)
                    if weights is not None:
                        weights = weights * bias_weights
                        weights /= weights.sum()
                    else:
                        weights = bias_weights
                # else: all-zero bias → fall back to existing weights

        # 3. Perform actual sampling
        sample_features = candidates[np.random.choice(len(candidates), size=self.num_samples_per_patch, replace=True, p=weights)]
        return sample_features

    def input_dim(self):
        """Returns number of features for each item"""
        return len(self.feature_names_list)
