import os

import numpy as np
import pandas as pd
from utils import NODATA, fire_cause_mapping, load_raster
from visualize import visualize_ignition_grid


def build_esc_fire_distribution_zone_mapping(esc_fire_distribution_file_path:str, cause: int, season: int) -> dict[int, float]:
    """ Load ignition distribution file and return zone to probability mapping for given cause and season"""
    esc_fire_dist = pd.read_csv(esc_fire_distribution_file_path)

    # convert percentage to probability
    esc_fire_dist["esc_fires"] = esc_fire_dist["esc_fires"] / 100.0
    esc_fire_dist_subset = esc_fire_dist[(esc_fire_dist["season"] == season) & (esc_fire_dist["cause"] == cause)].copy()
    
    # Convert it to probabilities over zones
    esc_fire_dist_subset["esc_fires"] = esc_fire_dist_subset["esc_fires"] / esc_fire_dist_subset["esc_fires"].sum()

    assert len(esc_fire_dist_subset) == len(esc_fire_dist_subset["zone"].tolist())
    
    # Map: zone_id -> probability
    esc_fire_zone_prob_mapping = dict(zip(esc_fire_dist_subset["zone"].astype(int), esc_fire_dist_subset["esc_fires"], strict=True))

    return esc_fire_zone_prob_mapping


def project_esc_fire_distribution_over_grid(zone_raster: np.ma.MaskedArray, esc_fire_zone_prob_mapping: dict[int, float]) -> np.ndarray:
    """ Project zone distribution probabilities over zone grid"""
    zones_data = zone_raster.data.astype("int32")
    zones_mask = zone_raster.mask

    esc_fire_prob_grid = np.full(zones_data.shape, fill_value=NODATA, dtype=np.float64)

    for z_id, z_w in esc_fire_zone_prob_mapping.items():
        valid_mask = ~zones_mask & (zones_data == z_id)
        if np.any(valid_mask):
            esc_fire_prob_grid[valid_mask] = z_w
    
    # size (H, W)
    return esc_fire_prob_grid

def sample_fire_density_grid(zone_grid_file_path: str, esc_fire_distribution_file_path:str, season: int, cause:int)-> np.ndarray:
    """ Build fire density grid based on probability of escaped fires per fire weather zone"""
    zone_raster = load_raster(zone_grid_file_path)
    
    esc_fire_zone_prob_mapping = build_esc_fire_distribution_zone_mapping(esc_fire_distribution_file_path=esc_fire_distribution_file_path, cause=cause, season=season)
    esc_fire_grid = project_esc_fire_distribution_over_grid(zone_raster=zone_raster, esc_fire_zone_prob_mapping=esc_fire_zone_prob_mapping)

    return esc_fire_grid

def load_ignition_grid(ignition_grids_folder_path: str, cause: int = None, season: int = None)-> np.ndarray:
    """Load ignition grids for a specific season/cause or all seasons/causes"""
    if season and cause:
        file_name = f"ign_s{season}_{fire_cause_mapping[cause]}.asc"
        ignition_raster = load_raster(os.path.join(ignition_grids_folder_path, file_name))

        return ignition_raster

    # else, loop over all ignition grids (across causes/seasons) and output one grid
    ignition_raster_files = [f for f in os.listdir(ignition_grids_folder_path) if f.endswith(".asc")]
    out_ignition_grids = []
    for file_name in ignition_raster_files:
        ignition_raster = load_raster(os.path.join(ignition_grids_folder_path, file_name))
        ignition_raster = ignition_raster.filled(NODATA)
        out_ignition_grids.append(ignition_raster)
    
    # size (H, W)
    out_ignition_grids = np.stack(out_ignition_grids, axis=0)
    max_ignition_grid = np.max(out_ignition_grids, axis=0)

    return max_ignition_grid

# TODO: delete later
if __name__ == "__main__":
    root_dir = "../yan_bp3/hex05"

    season = 1
    cause = 1

    out_ignition_grid = load_ignition_grid(ignition_grids_folder_path=os.path.join(root_dir, "ignitions_module/ignition_grids"), season=season, cause=cause)
    visualize_ignition_grid(out_ignition_grid, cause=cause, season=season)

    # optionally, we can also use this grid
    # out_fire_density_grid = sample_fire_density_grid(zone_grid_file_path=os.path.join(root_dir,"mapped_inputs/cfrs.asc"),
    #                          esc_fire_distribution_file_path=os.path.join(root_dir,"ignitions_module/Nb_ignitions_zone_season_cause_5.csv"),
    #                          season=season, cause=cause)
    # visualize_ignition_grid(out_fire_density_grid, season=season, cause=cause)
