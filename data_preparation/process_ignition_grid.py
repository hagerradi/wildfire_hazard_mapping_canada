import os
import re

import numpy as np
import pandas as pd
import rasterio
from utils import visualize_raster

cause_mapping = {"h": 1, "l": 2}


def load_raster(path: str)-> np.ndarray:
        """ Load raster from given path"""
        with rasterio.open(path) as src:
            raster = src.read(1, masked=True) # mask out the nodata (-9999 values)

        return raster


def load_ignition_distribution_zone_mapping(ignition_distribution_file_path, cause: int, season: int) -> dict[int: float]:
    """ Load ignition distribution file and return zone to probability mapping for given cause and season"""
    ignitions_dist = pd.read_csv(ignition_distribution_file_path)

    # convert percentage to probability
    ignitions_dist["esc_fires"] = ignitions_dist["esc_fires"] / 100.0
    ignitions_dist_subset = ignitions_dist[(ignitions_dist["season"] == season) & (ignitions_dist["cause"] == cause)].copy()
    
    # Convert it to probabilities over zones
    ignitions_dist_subset["esc_fires"] = ignitions_dist_subset["esc_fires"] / ignitions_dist_subset["esc_fires"].sum()

    assert len(ignitions_dist_subset) == len(ignitions_dist_subset["zone"].tolist())
    
    # Map: zone_id -> probability
    zone_prob_mapping = dict(zip(ignitions_dist_subset["zone"].astype(int), ignitions_dist_subset["esc_fires"], strict=True))
    print("Zone to prob mapping:", zone_prob_mapping)

    return zone_prob_mapping


def project_zone_distribution_over_grid(ignition_raster: np.ndarray, zone_raster: np.ndarray, zone_prob_mapping: dict[int: float])-> np.ndarray:
    """ Project zone distribution probabilities over ignition raster grid"""
    ignition_data   = ignition_raster.data.astype("float64")
    ignition_mask   = ignition_raster.mask

    zones_data = zone_raster.data.astype("int32")
    zones_mask = zone_raster.mask

    # valid where both rasters have data
    valid = (~ignition_mask) & (~zones_mask)

    # initialize result
    reweighted_raster = np.zeros_like(ignition_data, dtype="float64")

    for z_id, z_w in zone_prob_mapping.items():
        valid_mask = valid & (zones_data == z_id)
        if not np.any(valid_mask):
            continue
        
        # multiply cell weight by zone weight
        reweighted_raster[valid_mask] = ignition_data[valid_mask] * z_w

    return reweighted_raster

def build_ignition_conditional_prob_grid(ignition_grid_folder_path: str, zone_grid_file_path: str, ignition_distribution_file_path:str)-> list[np.ndarray]:
    """ Build ignition grids re-weighted by zone distribution probabilities"""
    zone_raster = load_raster(zone_grid_file_path)
    
    ignition_raster_files = [f for f in os.listdir(ignition_grid_folder_path) if f.endswith(".asc")]
    out_ignition_grids = []
    for file_name in ignition_raster_files:
        print(file_name)
        matched = re.match(r"ign_s(\d+)_(\w)\.asc", file_name)
        season = int(matched.group(1))
        cause = matched.group(2)
        ignition_raster = load_raster(os.path.join(ignition_grid_folder_path, file_name))

        zone_prob_mapping = load_ignition_distribution_zone_mapping(ignition_distribution_file_path=ignition_distribution_file_path, cause=cause_mapping[cause], season=season)
        reweighted_raster = project_zone_distribution_over_grid(ignition_raster=ignition_raster, zone_raster=zone_raster, zone_prob_mapping=zone_prob_mapping)

        out_ignition_grids.append(reweighted_raster)
        
        print(f"Processed re-weighted ignition grid for season {season} and cause {cause}")
        print(f"  Min: {np.min(reweighted_raster)}, Max: {np.max(reweighted_raster)}, Sum: {np.sum(reweighted_raster)}")
    

    return out_ignition_grids


# delete later
if __name__ == "__main__":
    root_dir = "/Users/hagerradi/Projects/wildfire/yan_bp3/hex05"

    out_ignition_grids = build_ignition_conditional_prob_grid(ignition_grid_folder_path=os.path.join(root_dir, "ignitions_module/ignition_grids"),
                                        zone_grid_file_path=os.path.join(root_dir,"mapped_inputs/cfrs.asc"),
                                        ignition_distribution_file_path=os.path.join(root_dir,"ignitions_module/Nb_ignitions_zone_season_cause_5.csv"))
    
    visualize_raster(out_ignition_grids[3], cause=2, season=2)
