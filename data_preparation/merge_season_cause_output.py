import argparse
import os
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from data_preparation.grid_loader.utils import denormalize_burn_count, get_range_burn_count
from data_preparation.paths import ESC_FIRE_DIST_PATH
from data_preparation.utils import find_hex_ids


def aggregate_burn_count_predictions(
    root_dir: str, hex_id: str, predictions_dir: str, min_count: float, max_count: float, output_suffix: str = "merged_predicted_bc.tif"
) -> None:
    """
    Merges predicted season-cause count maps into a single global count map.

    Args:
        root_dir (str): Data root (where hexel folders are).
        hex_id (str): Hex ID to process.
        predictions_dir (str): Folder containing the predicted scenario TIFs.
        min_count (float): min count for denormalization.
        max_count (float): max count for denormalization.
        output_suffix (str): Suffix of output file name to use.
    """
    print(f"--- Merging and denormalizing Hexel {hex_id}. ---")

    hex_dir = os.path.join(root_dir, f"hex{hex_id}")

    # get seasons and causes from the csv template
    csv_path = os.path.join(hex_dir, ESC_FIRE_DIST_PATH + str(int(hex_id)) + ".csv")
    if not os.path.exists(csv_path):
        return
    df = pd.read_csv(csv_path)
    seasons = df["season"].unique().tolist()
    causes = df["cause"].unique().tolist()

    total_counts = None
    profile = None
    files_merged = 0

    # main loop that aggregates counts from all predicted scenarios
    for season, cause in product(seasons, causes):
        s_str = str(season).replace(" ", "")
        c_str = str(cause).replace(" ", "")
        fname = f"hex_{hex_id}_season_{s_str}_cause_{c_str}_bc.tif"
        fpath = os.path.join(predictions_dir, fname)

        try:
            if os.path.exists(fpath):
                with rasterio.open(fpath) as src:
                    norm_data = src.read(1)

                    if profile is None:
                        profile = src.profile.copy()
                        total_counts = np.zeros_like(norm_data, dtype="float64")

                    # denormalize counts based on min and max values
                    real_counts = denormalize_burn_count(norm_data, min_count, max_count)

                    total_counts += real_counts
                    files_merged += 1
            else:
                pass

        except Exception as e:
            print(f"Error loading {fname}: {e}")
            continue

    # save the merged counts map
    # TODO: this needs more thought, since we are not able to get the total counts to match the
    # target. Potentially, we can add clamping here.
    if total_counts is not None and files_merged > 0:
        final_filename = f"hex_{hex_id}_{output_suffix}"
        out_path = os.path.join(predictions_dir, final_filename)

        final_int_counts = np.rint(total_counts).astype("int32")
        profile.update(dtype="int32", compress="lzw", nodata=-9999)  # type: ignore

        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(final_int_counts, 1)

        print(f"Saved Global Count Map: {out_path}.")
        print(f"Scenarios Merged: {files_merged}.")
    else:
        print("Failed to generate map.")


def main():
    """Script that loops over hexel folders, gets the predicted scenarios counts, denormalizes them, and merges them into the final count map."""
    parser = argparse.ArgumentParser(description="Merge predicted season-cause count maps into a global map.")
    parser.add_argument("--root_dir", type=str, required=True, help="Original data root (contains hex folders).")
    parser.add_argument("--predictions_dir", type=str, required=True, help="Folder containing the predicted season-cause TIFs.")
    parser.add_argument("--output_suffix", type=str, default="merged_predicted_bc.tif", help="Suffix for output file.")

    args = parser.parse_args()

    # get min and max counts for denormalization
    max_count, min_count = get_range_burn_count(args.root_dir)

    # process the hexels
    hex_ids = find_hex_ids(args.root_dir)

    # main loop to merge scenario maps into the final predicted map
    for hex_id in hex_ids:
        check_pattern = f"hex_{hex_id}_*"
        if not list(Path(args.predictions_dir).glob(check_pattern)):
            continue
        aggregate_burn_count_predictions(args.root_dir, hex_id, args.predictions_dir, min_count, max_count, args.output_suffix)


if __name__ == "__main__":
    main()
