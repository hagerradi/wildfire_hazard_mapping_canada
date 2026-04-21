from __future__ import annotations

import json
import os

import numpy as np

from data_preparation.paths import Paths
from data_preparation.spatial import NODATA, load_fuel_grid, load_ignition_grid, load_spatial_raster
from data_preparation.utils import feature_names


def get_num_channels_array(arr: np.ndarray) -> int:
    """Returns the number of channels a particular feature will take"""
    if len(arr.shape) == 2:
        return 1
    return arr.shape[-1]


def generate_feature_channel_map(feature_list: list[np.ndarray], feature_channel_map_path: str):
    """Maps the feature names to the corresponding channels in our input stack"""
    feature_channel_map = dict()
    channel = 0
    for i, feature in enumerate(feature_list):
        feature_channels = get_num_channels_array(feature)
        feature_channel_map[feature_names[i]] = list(range(channel, channel + feature_channels))
        channel += feature_channels
    os.makedirs(os.path.dirname(feature_channel_map_path), exist_ok=True)
    with open(feature_channel_map_path, "w") as f:
        json.dump(feature_channel_map, f, indent=4)


def load_spatial_features_per_hexel(
    root_dir: str,
    hex_id: str,
    feature_channel_map_path: str,
    modelling_approach: int = 1,
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, tuple[int, int]] | None]:
    """
    Load all data (features and output) per hexel
    root_dir: Root directory containing all hexels.
    hex_id: Hexel id to load.
    modelling_approach: 1 for joint season-cause modelling, 2 for separate season-cause modelling.
    Returns:
        all_features: np.ndarray of shape (N, H, W, num_features)
        all_masks: np.ndarray of shape (N, H, W)
        season_cause_mapping: dict mapping index to (season, cause)
    """

    def stack_sample(
        fuel_grid: np.ma.MaskedArray,
        elevation_grid: np.ma.MaskedArray,
        ignition_grid: np.ma.MaskedArray,
        bp_out_grid: np.ma.MaskedArray,
        fi_out_grid: np.ma.MaskedArray,
        ros_out_grid: np.ma.MaskedArray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack all features and compute mask."""
        features_list = [
            fuel_grid[:, :, np.newaxis],
            elevation_grid[:, :, np.newaxis],
            ignition_grid[:, :, np.newaxis],
            bp_out_grid[:, :, np.newaxis],
            fi_out_grid[:, :, np.newaxis],
            ros_out_grid[:, :, np.newaxis],
        ]
        if not os.path.exists(feature_channel_map_path):
            generate_feature_channel_map(features_list, feature_channel_map_path)
        stacked = np.concatenate(
            features_list,
            axis=-1,
        )
        stacked = np.ma.filled(stacked, fill_value=NODATA).astype(np.float32)
        # Get all the masks for all the season/cause and channels
        all_feat_mask = np.isnan(stacked)
        # Aggregate the channel masks to create a single mast (OR operation)
        mask = np.any(all_feat_mask, axis=-1)
        # Redo the feats with the new mask
        stacked[mask] = NODATA
        if int(np.sum(mask.astype(bool) != np.isnan(fuel_grid).astype(bool))) > 0:
            print("======The fuel mask is not the same as the cumulative mask=====")
        return stacked, mask

    # identify all seasons and causes first
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    # load all common grids
    fuel_grid = load_fuel_grid(root_dir=root_dir, hex_id=hex_id)  # noqa: F821

    elevation_grid, _ = load_spatial_raster(
        path=all_paths.elevation_grid(hex_id=hex_id), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id)
    )

    if modelling_approach == 1:
        # input
        ignition_grid = load_ignition_grid(root_dir=root_dir, hex_id=hex_id)

        bp_out_grid, _ = load_spatial_raster(all_paths.output_burn_prob(), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id))
        fi_out_grid, _ = load_spatial_raster(all_paths.output_fire_intensity(), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id))
        ros_out_grid, _ = load_spatial_raster(all_paths.output_ros(), actual_mask_path=all_paths.mask_grid_actual(hex_id=hex_id))

        stacked_features, mask = stack_sample(fuel_grid, elevation_grid, ignition_grid, bp_out_grid, fi_out_grid, ros_out_grid)
        return np.expand_dims(stacked_features, axis=0), np.expand_dims(mask, axis=0), None

    # modelling approach 2
    raise ValueError("Data Season mapping not supported yet!")


if __name__ == "__main__":
    out, mask, _ = load_spatial_features_per_hexel(
        root_dir="../burnp3plus", hex_id="01", feature_channel_map_path="../burnp3plus/feature_channel_map.json"
    )
