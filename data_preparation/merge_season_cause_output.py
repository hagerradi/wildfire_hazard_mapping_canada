import argparse
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from data_preparation.generate_season_cause_output import FireCountRasterizer
from data_preparation.grid_loader.utils import load_fire_shapefiles
from data_preparation.paths import ESC_FIRE_DIST_PATH
from data_preparation.utils import find_hex_ids


def aggregate_predictions(root_dir: str, hex_id: str, predictions_dir: str, output_filename: str = "global_map_bp.tif") -> None:
    """
    Args:
        root_dir: Data root (to find shapefiles for weighting).
        hex_id: Hex ID.
        predictions_dir: Folder containing the stitched predicted TIFs.
        output_filename: Name of the final global file.
    """
    print(f"--- Merging Hexel {hex_id} ---")

    hex_dir = os.path.join(root_dir, f"hex{hex_id}")

    try:
        shp_paths = load_fire_shapefiles(hex_dir)
        rasterizer = FireCountRasterizer(shp_paths, template_raster_path=None)
    except Exception as e:
        print(f"Skipping {hex_id} (Shapefile Load Error): {e}")
        return

    # get seasons and causes
    csv_path = os.path.join(hex_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv")
    if not os.path.exists(csv_path):
        print(f"Skipping {hex_id}: CSV not found.")
        return

    df = pd.read_csv(csv_path)
    seasons = df["season"].unique().tolist()
    causes = df["cause"].unique().tolist()

    accumulated_prob = None
    total_iterations = 0
    profile = None

    for season, cause in product(seasons, causes):
        _, n_iters = rasterizer.compute_counts(season=season, cause=cause)

        if n_iters == 0:
            continue

        fname = f"hex_{hex_id}_{season}_{cause}_bp.tif".replace(" ", "")
        fpath = os.path.join(predictions_dir, fname)

        try:
            if os.path.exists(fpath):
                with rasterio.open(fpath) as src:
                    data = src.read(1)
                    if profile is None:
                        profile = src.profile.copy()
                        accumulated_prob = np.zeros_like(data, dtype="float64")

                    accumulated_prob += data * n_iters
                    total_iterations += n_iters
                    print(f"  + Merging: {season} - {cause} (Weight: {n_iters})")
            else:
                print(f"  Warning: File missing for valid iterations ({n_iters}): {fname}")
                total_iterations += n_iters

        except Exception as e:
            print(f"  Error loading {fname}: {e}")
            continue

    if accumulated_prob is not None and total_iterations > 0:
        global_map = accumulated_prob / total_iterations

        out_folder = os.path.join(predictions_dir, "global_merged")
        os.makedirs(out_folder, exist_ok=True)
        out_path = os.path.join(out_folder, output_filename)

        profile.update(dtype="float32", compress="lzw")  # type: ignore

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(global_map.astype("float32"), 1)

        print(f"Saved Global Merged Map: {out_path}")
        print(f"Total Aggregated Iterations: {total_iterations}")
    else:
        print("Failed to generate global map (No valid data found).")


def main():
    parser = argparse.ArgumentParser(description="Merge season-cause specific maps into a global probability map.")
    parser.add_argument("--root_dir", type=str, required=True, help="Original data root (contains hex folders/shapefiles).")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Folder containing the stitched season-cause TIFs.")
    parser.add_argument("--output_name", type=str, default="global_map_bp.tif", help="Filename for the merged output.")

    args = parser.parse_args()

    hex_ids = find_hex_ids(args.root_dir)

    for hex_id in hex_ids:
        check_pattern = f"hex_{hex_id}_*"
        if not list(Path(args.predictions_dir).glob(check_pattern)):
            continue

        aggregate_predictions(args.root_dir, hex_id, args.predictions_dir, args.output_name)


if __name__ == "__main__":
    main()
