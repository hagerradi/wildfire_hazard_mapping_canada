"""
Script for aggregating and preparing the sequential weather table
and processing the fire size distribution table.
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Callable, Optional

import pandas as pd
import numpy as np

from data_preparation.feature_processing.weather import load_weather_list, preprocess_weather_list
from data_preparation.utils import aggregate_csv_by_pattern, process_fire_size_df



def main():
    parser = argparse.ArgumentParser(description="Compile raw weather CSVs into a single lookup table and processing fire size distribution table.")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to raw data root directory")
    parser.add_argument("--modelling_approach", type=str, required=True, help="String '1' or '2' indicating modelling approach used before these steps", default="1")
    parser.add_argument("--weather_output_file", type=str, help="File name (ends in .csv) to be used to save the aggregated and procssed weather table", default="weather_table.csv")
    parser.add_argument("--fire_size_input_file", type=str, help="File name (ends in .csv) of existing fire size distribution table", default="df_fire_fru.csv")
    parser.add_argument("--fire_size_output_file", type=str, help="File name (ends in .csv) to be used to save the processed distribution table", default="df_fire_fru_processed.csv")
    parser.add_argument("--save_dir", type=str, help="Path to save directory (optional, used for testing)", default=None)

    args = parser.parse_args()
    root_path = Path(args.root_dir)
    input_dir = Path(args.root_dir) / f"data_samples_approach_{args.modelling_approach}_weather"
    output_dir = Path(args.save_dir) if args.save_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    
    weather_save_path = output_dir / args.weather_output_file
    fire_size_save_path = output_dir / args.fire_size_output_file
    fire_size_read_path = input_dir / args.fire_size_input_file

    print(f"Starting weather table build...")
    pattern = "hex*/burning_conditions_module/hex_*_weather*.csv"  # Accounts for all types of naming including anomalies
    df_weather = aggregate_csv_by_pattern(root_dir=root_path, pattern=pattern, load_function=load_weather_list)
    # Apply weather specific global normalization
    print("Applying global preprocessing...")
    df_weather = preprocess_weather_list(df_weather)
    df_weather.to_csv(weather_save_path, index=False)
    print(f"Successfully saved to {weather_save_path}")

    print(f"Starting fire size distribution processing...")
    df_fire_size = pd.read_csv(fire_size_read_path)
    df_fire_size_processed = process_fire_size_df(df_fire_size)
    df_fire_size_processed.to_csv(fire_size_save_path)
    print(f"Successfully save to {fire_size_save_path}")


if __name__ == "__main__":
    main()
