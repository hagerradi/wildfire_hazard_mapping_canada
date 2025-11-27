import os
from itertools import product

import numpy as np
import pandas as pd

from data_preparation.grid_loading import (
    load_elevation_grid,
    load_fire_density_grid,
    load_fuel_grid,
    load_ignition_grid,
    load_output_burn_prob_grid,
    load_weather_grid,
    load_wind_grid,
)
from data_preparation.utils import (
    ELEVATION_GRID_PATH,
    ESC_FIRE_DIST_PATH,
    FIRE_ZONE_GRID_PATH,
    FUEL_GRID_PATH,
    FUEL_TABLE_PATH,
    IGNITION_PROB_PATH,
    OUTPUT_BURN_PROB_PATH,
    WEATHER_LIST_PATH,
    WIND_GRID_DIR_PATH,
)


def load_features_per_hexel(root_dir: str, hex_id: str):
    # identify all seasons and causes first
    root_dir = os.path.join(root_dir, "hex" + str(hex_id))
    ignitions_df = pd.read_csv(os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"))

    seasons = ignitions_df["season"].unique().tolist()
    causes = ignitions_df["cause"].unique().tolist()
    
    all_features = []
    all_masks = []
    for season, cause in product(seasons, causes):
        # input
        ignition_prob = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, IGNITION_PROB_PATH), season=season, cause=cause)
        
        esc_fires_prob = load_fire_density_grid(zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH),
                             esc_fire_distribution_file_path=os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"),
                             season=season, cause=cause)
        
        fuel_grid = load_fuel_grid(path=os.path.join(root_dir, FUEL_GRID_PATH),
                                fuel_table_path=os.path.join(root_dir, FUEL_TABLE_PATH))  # noqa: F821
        
        elevation_grid = load_elevation_grid(path=os.path.join(root_dir, ELEVATION_GRID_PATH))  # noqa: F821
        
        weather_grid = load_weather_grid(weather_list_file_path=os.path.join(root_dir, WEATHER_LIST_PATH + f"/hex_{hex_id}_weather_list.csv"),
                        zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH), 
                        season=season)

        wind_grid = load_wind_grid(path = os.path.join(root_dir, WIND_GRID_DIR_PATH))

        # output
        out_burn_prob = load_output_burn_prob_grid(path = os.path.join(root_dir, OUTPUT_BURN_PROB_PATH + f"/hex_{hex_id}_20000_iter_bp.tif"))

        # stack them all.
        stacked_features = np.concatenate([ignition_prob[:, :, np.newaxis],
                                        esc_fires_prob[:, :, np.newaxis],
                                        fuel_grid[:, :, np.newaxis],
                                        elevation_grid[:, :, np.newaxis],
                                        weather_grid,
                                        wind_grid,
                                        out_burn_prob[:, :, np.newaxis]], axis=-1)
        
        mask = np.isnan(elevation_grid)

        all_features.append(stacked_features)
        all_masks.append(mask)
    
    # features: (N, H, W, 36), mask: (N, H, W)
    return np.stack(all_features), np.stack(all_masks)  

# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3"
    hex_ids = ["05", "10", "16"]
    
    all_features, all_masks = load_features_per_hexel(root_dir=root_dir, hex_id=hex_ids[0])
    print(all_features.shape)
    print(all_masks.shape)