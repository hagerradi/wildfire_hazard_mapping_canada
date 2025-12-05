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
    load_weather_grid,
    load_wind_grid,
)
from data_preparation.grid_loader.output import load_output_burn_count_grid, load_output_burn_prob_grid
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
from data_preparation.utils import find_fire_output_file


def load_features_per_hexel(root_dir: str, hex_id: str, modelling_approach: int = 1, output_type: str = "prob") -> tuple[np.ndarray, np.ndarray, dict[int, tuple[int, int]]]:
    """
    Load all data (features and output) per hexel
    root_dir: root directory containing all hexels
    hex_id: hexel id to load
    modelling_approach: 1 for joint season-cause modelling, 2 for separate season-cause modelling
    output_type (str): 'count' for counts, 'prob' for probs
    Returns:
        all_features: np.ndarray of shape (N, H, W, num_features)
        all_masks: np.ndarray of shape (N, H, W)
        season_cause_mapping: dict mapping index to (season, cause)
    """
    # identify all seasons and causes first
    root_dir = os.path.join(root_dir, "hex" + str(hex_id))

    # load all common grids
    fuel_grid = load_fuel_grid(path=os.path.join(root_dir, FUEL_GRID_PATH),
                        fuel_table_path=os.path.join(root_dir, FUEL_TABLE_PATH))  # noqa: F821

    elevation_grid = load_elevation_grid(path=os.path.join(root_dir, ELEVATION_GRID_PATH))  # noqa: F821

    wind_grid = load_wind_grid(path = os.path.join(root_dir, WIND_GRID_DIR_PATH))

    weather_list_file_path = str(list(Path(os.path.join(root_dir, WEATHER_LIST_PATH)).glob("*weather_list*.csv"))[0])
    
    def stack_sample(
        ignition_prob_grid: np.ndarray,
        esc_fires_prob_grid: np.ndarray,
        weather_grid: np.ndarray,
        out_grid: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stack all features and compute mask."""
        stacked = np.concatenate(
            [
                ignition_prob_grid[:, :, np.newaxis],
                esc_fires_prob_grid[:, :, np.newaxis],
                fuel_grid[:, :, np.newaxis],
                elevation_grid[:, :, np.newaxis],
                weather_grid,
                wind_grid,
                out_grid[:, :, np.newaxis],
            ],
            axis=-1,
        )
        mask = np.isnan(elevation_grid)
        return stacked, mask
    
    def load_output_grid(path):
        if output_type == "count":
            return load_output_burn_count_grid(path)
        elif output_type == "prob":
            return load_output_burn_prob_grid(path)
        else:
            raise ValueError(f"Invalid output_type: {output_type}.")

    if modelling_approach == 1:
        season_cause_mapping = None
        # input
        ignition_prob_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, IGNITION_PROB_PATH))
         
        esc_fires_prob_grid = load_fire_density_grid(zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH),
                             esc_fire_distribution_file_path=os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"))

        weather_grid = load_weather_grid(weather_list_file_path=weather_list_file_path,
                        zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH))
        
        # for approach 1, we use the existing raster output
        fpath = find_fire_output_file(root_dir, hex_id, output_type, season=None, cause=None)
        out_grid = load_output_grid(fpath)

        stacked_features, mask = stack_sample(
            ignition_prob_grid, esc_fires_prob_grid, weather_grid, out_grid
        )

        return np.expand_dims(stacked_features, axis=0), np.expand_dims(mask, axis=0), None

    # modelling approach 2
    ignitions_df = pd.read_csv(os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"))

    seasons = ignitions_df["season"].unique().tolist()
    causes = ignitions_df["cause"].unique().tolist()

    all_features = []
    all_masks = []
    season_cause_mapping = {}

    for i, season_cause in enumerate(product(seasons, causes)):
        season, cause = season_cause
        # input
        ignition_prob_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, IGNITION_PROB_PATH), season=season, cause=cause)
        
        esc_fires_prob_grid = load_fire_density_grid(zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH),
                             esc_fire_distribution_file_path=os.path.join(root_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"),
                             season=season, cause=cause)


        weather_grid = load_weather_grid(weather_list_file_path=weather_list_file_path,
                        zone_grid_file_path=os.path.join(root_dir, FIRE_ZONE_GRID_PATH), 
                        season=season)
        
        # for approach 2, we use the season-cause rasters
        fpath = find_fire_output_file(root_dir, hex_id, output_type, season=season, cause=cause)
        out_grid = load_output_grid(fpath)

        # stack them all.
        stacked_features, mask = stack_sample(
            ignition_prob_grid, esc_fires_prob_grid, weather_grid, out_grid
        )
        
        all_features.append(stacked_features)
        all_masks.append(mask)
        season_cause_mapping[i] = season_cause

    return np.stack(all_features), np.stack(all_masks), season_cause_mapping

# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3"
    hex_ids = ["05", "10", "16"]
    
    all_features, all_masks, season_cause_mapping = load_features_per_hexel(root_dir=root_dir,
                                                                            hex_id=hex_ids[0],
                                                                            modelling_approach=1,
                                                                            output_type="count")
    print(all_features.shape)
    print(all_masks.shape)
    print(season_cause_mapping)