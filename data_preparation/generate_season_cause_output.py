import os
from itertools import product
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import MergeAlg, rasterize

from data_preparation.grid_loader.utils import load_fire_shapefiles
from data_preparation.paths import ESC_FIRE_DIST_PATH, OUTPUT_BURN_PROB_PATH


class FireCountRasterizer:
    """Rasterization class to accumulate fire polygons from shapefiles."""

    def __init__(self, shp_paths: list[Path], template_raster_path: str):
        """Init. reading of shapefiles once."""
        with rasterio.open(template_raster_path) as src:
            self.out_shape = (src.height, src.width)
            self.transform = src.transform

        gdfs = []
        print(f"Using {len(shp_paths)} shp files...")
        for run_id, p in enumerate(shp_paths, start=1):
            g = gpd.read_file(p)[["season", "cause", "iteration", "geometry"]]
            g["run_id"] = run_id
            gdfs.append(g)

        if not gdfs:
            self.g_all = gpd.GeoDataFrame()
        else:
            self.g_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True), crs=gdfs[0].crs)

    def compute_counts(self, season: str = None, cause: str = None) -> tuple[np.ndarray, int]:
        """Method to accumulate the fire polygons."""

        if self.g_all.empty:
            return np.zeros(self.out_shape, dtype="int32"), 0

        df = self.g_all
        if season:
            df = df[df["season"] == season]
        if cause:
            df = df[df["cause"] == cause]

        if df.empty:
            return np.zeros(self.out_shape, dtype="int32"), 0

        # keeping only unique iterations
        iters = df[["run_id", "iteration"]].drop_duplicates().to_records(index=False)
        num_unique_iterations = len(iters)

        count_grid = np.zeros(self.out_shape, dtype="int32")

        # main raster loop
        for run_id, iter_id in iters:
            one_iter = df[(df["run_id"] == run_id) & (df["iteration"] == iter_id)]
            fire_polygons = ((geom, 1) for geom in one_iter.geometry)

            iter_raster = rasterize(
                shapes=fire_polygons,
                out_shape=self.out_shape,
                transform=self.transform,
                fill=0,
                all_touched=False,
                merge_alg=MergeAlg.replace,
                dtype="int16",
            )
            count_grid += iter_raster

        return count_grid, num_unique_iterations


def save_raster(data: np.ndarray, template_profile: dict[str, Any], out_path: str) -> None:
    """Saves a given ndarray to a GeoTIFF based on type.

    Args:
        data (np.ndarray): 2D array to save.
        template_profile (dict[str, Any]): Rasterio profile dictionary from the template file.
        out_path (str): The output file path.

    """
    profile = template_profile.copy()

    # for counts (int)
    if np.issubdtype(data.dtype, np.integer):
        profile.update(dtype="int32", nodata=-9999, compress="lzw")
    # for probs (float)
    else:
        profile.update(dtype="float32", compress="lzw")

    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(data, 1)
    print(f"Saved: {os.path.basename(out_path)}")


def generate_season_cause_burn_count_rasters(root_dir: str, hex_id: str) -> None:
    """
    Main function to generate and save list of raster files for a given hexel.

    Args:
        root_dir (str): The root dir.
        hex_id (str): The hexel ID.

    """
    print(f"--- Processing hexel {hex_id} ---")
    hex_dir = os.path.join(root_dir, f"hex{hex_id}")
    outputs_dir = os.path.join(hex_dir, OUTPUT_BURN_PROB_PATH)

    # get template for counts
    template_bc = next(Path(outputs_dir).glob("*_bc.tif"), None)
    if not template_bc:
        print(f"Skipping hexel {hex_id}: no template tif for counts found.")
        return

    # get template for probs
    template_bp = next(Path(outputs_dir).glob("*_bp.tif"), None)
    if not template_bp:
        print(f"Skipping hexel {hex_id}: no template tif for probs found.")
        return

    with rasterio.open(str(template_bc)) as src:
        profile_bc = src.profile
    with rasterio.open(str(template_bp)) as src:
        profile_bp = src.profile

    try:
        shp_paths = load_fire_shapefiles(hex_dir)
        rasterizer = FireCountRasterizer(shp_paths, str(template_bc))
    except Exception as e:
        print(f"Skipping {hex_id}: {e}")
        return

    df = pd.read_csv(os.path.join(hex_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv"))
    seasons = df["season"].unique().tolist()
    causes = df["cause"].unique().tolist()

    # main loop for season-cause rasterization
    for season, cause in product(seasons, causes):
        print(f"Generating: season {season} - cause {cause} ...")

        count_grid, num_iters = rasterizer.compute_counts(season=season, cause=cause)

        prob_grid = count_grid.astype("float32") / num_iters if num_iters > 0 else np.zeros_like(count_grid, dtype="float32")

        fname_bp = f"hex_{hex_id}_season_{season}_cause_{cause}_bp.tif".replace(" ", "")
        fname_bc = f"hex_{hex_id}_season_{season}_cause_{cause}_bc.tif".replace(" ", "")

        save_raster(count_grid, profile_bc, os.path.join(outputs_dir, fname_bc))
        save_raster(prob_grid, profile_bp, os.path.join(outputs_dir, fname_bp))


if __name__ == "__main__":
    root_dir = "../yan_bp3"
    hex_ids = ["05", "10", "16"]

    generate_season_cause_burn_count_rasters(root_dir, hex_ids[0])
