import glob
import json
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.grid_loader import (
    load_elevation_grid,
    load_fire_density_grid,
    load_fuel_grid,
    load_ignition_grid,
    load_output_burn_grid,
    load_weather_grid,
    load_wind_grid,
)
from data_preparation.grid_loader.utils import NODATA
from data_preparation.paths import (
    ELEVATION_GRID_PATH,
    ESC_FIRE_DIST_PATH,
    FIRE_ZONE_GRID_PATH,
    FUEL_GRID_PATH,
    FUEL_TABLE_PATH,
    IGNITION_PROB_PATH,
    WEATHER_LIST_PATH,
    WIND_GRID_DIR_PATH,
)
from data_preparation.utils import feature_names, find_simulation_output_file


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


def load_features_per_hexel(
    root_dir: str, hex_id: str, feature_channel_map_path: str, modelling_approach: int = 2, output_type: str = "count", weather_sampling: str = "dist",
) -> tuple[np.ndarray | None, np.ndarray | None, dict[int, tuple[int, int]] | None]:
    """
    Load all data (features and output) per hexel
    root_dir: Root directory containing all hexels.
    hex_id: Hexel id to load.
    modelling_approach: 1 for joint season-cause modelling, 2 for separate season-cause modelling.
    output_type (str): Type of fire output to use. Use 'count' for fire counts, or 'prob' for probs normalized by unique iterations.
    Returns:
        all_features: np.ndarray of shape (N, H, W, num_features)
        all_masks: np.ndarray of shape (N, H, W)
        season_cause_mapping: dict mapping index to (season, cause)
    """
    fire_output_types = ["count", "prob"]  # for now we only support 2 types of outputs
    if output_type not in fire_output_types:
        raise ValueError(f"Invalid output_type: '{output_type}'. Must be one of {fire_output_types}.")

    # identify all seasons and causes first
    root_dir = os.path.join(root_dir, "hex" + str(hex_id))

    # load all common grids
    fuel_grid = load_fuel_grid(path=os.path.join(root_dir, FUEL_GRID_PATH), fuel_table_path=os.path.join(root_dir, FUEL_TABLE_PATH))  # noqa: F821

    elevation_grid = load_elevation_grid(path=os.path.join(root_dir, ELEVATION_GRID_PATH))  # noqa: F821

    wind_grid = load_wind_grid(path=os.path.join(root_dir, WIND_GRID_DIR_PATH))

    weather_list_file_path = str(list(Path(os.path.join(root_dir, WEATHER_LIST_PATH)).glob("*weather_list*.csv"))[0])

    def stack_sample(
        ignition_prob_grid: np.ndarray,
        esc_fires_prob_grid: np.ndarray,
        weather_grid: np.ndarray,
        out_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack all features and compute mask."""
        features_list = [
            ignition_prob_grid[:, :, np.newaxis],
            esc_fires_prob_grid[:, :, np.newaxis],
            fuel_grid[:, :, np.newaxis],
            elevation_grid[:, :, np.newaxis],
            weather_grid,
            wind_grid,
            out_grid[:, :, np.newaxis],
        ]
        if not os.path.exists(feature_channel_map_path):
            generate_feature_channel_map(features_list, feature_channel_map_path)
        stacked = np.concatenate(
            features_list,
            axis=-1,
        )
        # Get all the masks for all the season/cause and channels
        all_feat_mask = np.isnan(stacked)
        # Aggregate the channel masks to create a single mast (OR operation)
        mask = np.any(all_feat_mask, axis=-1)
        # Redo the feats with the new mask
        stacked[mask] = NODATA
        if int(np.sum(mask.astype(bool) != np.isnan(elevation_grid).astype(bool))) > 0:
            print("======The elevation mask is not the same as the cumulative mask=====")
        return stacked, mask

    def load_output_grid(path):
        if not os.path.exists(path):
            dtype = "int32" if output_type == "count" else "float32"
            return np.zeros_like(elevation_grid, dtype=dtype)  # in case no fires for a scenario
        return load_output_burn_grid(path)

    pattern = os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + "*.csv")
    esc_fire_distribution_file_paths = glob.glob(pattern)
    if len(esc_fire_distribution_file_paths) > 0:
        esc_fire_distribution_file_path = esc_fire_distribution_file_paths[0]
    else:
        print(f"The file {pattern} does not exist")
        return None, None, None

    if modelling_approach == 1:
        season_cause_mapping = None
        # input
        ignition_prob_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, IGNITION_PROB_PATH))

        esc_fires_prob_grid = load_fire_density_grid(
            zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH),
            esc_fire_distribution_file_path=esc_fire_distribution_file_path,
        )

        weather_grid = load_weather_grid(
            weather_list_file_path=weather_list_file_path, zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH), sampling=weather_sampling
        )

        # for approach 1, we use the existing raster output
        # ASSUMPTION: we only support probability for approach 1
        fpath = find_simulation_output_file(root_dir, hex_id, output_type="prob", season=None, cause=None)
        out_grid = load_output_grid(fpath)

        stacked_features, mask = stack_sample(ignition_prob_grid, esc_fires_prob_grid, weather_grid, out_grid)
        return np.expand_dims(stacked_features, axis=0), np.expand_dims(mask, axis=0), None

    # modelling approach 2
    ignitions_df = pd.read_csv(esc_fire_distribution_file_path)

    seasons = ignitions_df["season"].unique().tolist()
    causes = ignitions_df["cause"].unique().tolist()

    all_features = []
    all_masks = []
    season_cause_mapping = {}

    for i, season_cause in enumerate(product(seasons, causes)):
        season, cause = season_cause
        # input
        ignition_prob_grid = load_ignition_grid(
            ignition_grids_folder_path=os.path.join(root_dir, IGNITION_PROB_PATH), season=season, cause=cause
        )

        esc_fires_prob_grid = load_fire_density_grid(
            zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH),
            esc_fire_distribution_file_path=esc_fire_distribution_file_path,
            season=season,
            cause=cause,
        )

        weather_grid = load_weather_grid(
            weather_list_file_path=weather_list_file_path, zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH), season=season
        )

        # for approach 2, we use the season-cause rasters
        fpath = find_simulation_output_file(root_dir, hex_id, output_type, season=season, cause=cause)
        out_grid = load_output_grid(fpath)

        # stack them all.
        stacked_features, mask = stack_sample(ignition_prob_grid, esc_fires_prob_grid, weather_grid, out_grid)

        all_features.append(stacked_features)
        all_masks.append(mask)
        season_cause_mapping[i] = season_cause

    return np.stack(all_features), np.stack(all_masks), season_cause_mapping
