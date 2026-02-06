import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd

from data_preparation.feature_processing.weather import load_weather_list, preprocess_weather_list
from data_preparation.utils import aggregate_csv_by_pattern


def main():
    parser = argparse.ArgumentParser(description="Compile raw weather CSVs into a single lookup table")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to raw data root directory")
    parser.add_argument("--file_name", type=str, help="File name (ends in .csv)", default="weather_table.csv")
    parser.add_argument("--save_dir", type=str, help="Path to save directory (optional)", default=None)

    args = parser.parse_args()
    root_path = Path(args.root_dir)
    file_name = args.file_name
    if args.save_dir:
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    else:
        save_dir = root_path
    save_file_path = save_dir / file_name

    print(f"Starting weather table build...")
    pattern = "hex*/burning_conditions_module/hex_*_weather*"  # Accounts for all types of naming including anomalies
    df = aggregate_csv_by_pattern(root_dir=root_path, pattern=pattern, load_function=load_weather_list)
    # Apply weather specific global normalization
    print("Applying global preprocessing...")
    df = preprocess_weather_list(df)
    df.to_csv(save_file_path, index=False)
    print(f"Successfully saved to {save_file_path}")


if __name__ == "__main__":
    main()
