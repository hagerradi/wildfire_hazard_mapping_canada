import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Optional

import pandas as pd
from data_preparation.feature_processing.utils import check_weather_list

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# We keep this list to enforce numeric types on the measurement columns
# (check_weather_list handles 'season' and 'wx_zone', but not 'temp', 'rh', etc.)
NUMERIC_COLS = [
    'temp', 'rh', 'ws', 'wd', 'prec', 
    'ffmc', 'dmc', 'dc', 'isi', 'bui', 'fwi'
]

def load_and_process_single_csv(file_path: Path) -> Optional[pd.DataFrame]:
    """
    Loads a single CSV, uses shared utils to standardize it, and extracts metadata.
    Returns None if file is empty or corrupted.
    """
    try:
        df = pd.read_csv(file_path)
        
        # This handles:
        # 1. Renaming 'WeatherZone' -> 'wx_zone', 'Temperature' -> 'temp', etc.
        # 2. Extracting integers from strings like "fru102" -> 102
        try:
            df = check_weather_list(df)
        except ValueError as e:
            logger.warning(f"Skipping {file_path.name}: {e}")
            return None

        df = df.loc[:, ~df.columns.str.startswith('Unnamed:')]  # Drop unnamed column
        
        try:
            # Assumes structure: .../hex*/burning_conditions_module/hex_*_weather_list.csv 
            hex_folder = file_path.parents[1].name              # e.g. hex05/
            hex_id = hex_folder.replace("hex", "")              # e.g. 05
            df["hex"] = hex_id                                  # Create column to identify hex
        except IndexError:
            logger.warning(f"Could not infer hex_id from path: {file_path}")
            return None
            
        return df

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return None

def build_aggregated_weather_dataset(root_dir: Path) -> pd.DataFrame:
    """
    Orchestrates the finding, loading, and merging of all weather files.
    """
    # Use glob to find files (generator is memory efficient)
    pattern = "hex*/burning_conditions_module/hex_*_weather_list.csv"
    files = sorted(root_dir.glob(pattern))
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {root_dir}")
    
    logger.info(f"Found {len(files)} weather files to process.")
    
    # Combine all weather tables
    data_frames = []
    for f in files:
        df = load_and_process_single_csv(f)
        if df is not None:
            data_frames.append(df)
            
    if not data_frames:
        raise ValueError("All files failed to load")
        
    full_df = pd.concat(data_frames, ignore_index=True)

    # Post process combined weather tables
    logger.info("Enforcing numeric types on measurement columns...")
    # We only need to handle measurements; season/zone were cast to int by check_weather_list
    for col in NUMERIC_COLS:
        if col in full_df.columns:
            full_df[col] = pd.to_numeric(full_df[col], errors='coerce')

    return full_df

def main():
    parser = argparse.ArgumentParser(description="Compile raw weather CSVs into a single lookup table")
    parser.add_argument("--root_dir", type=str, required=True, help="Path to raw data root directory")
    parser.add_argument("--file_name", type=str, help="File name (ends in .csv)", default="weather_table.csv")
    parser.add_argument("--output_dir", type=str, help="Path to save directory (optional)", default=None)

    args = parser.parse_args()
    root_path = Path(args.root_dir)
    file_name = args.file_name
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        output_dir = root_path
        
    save_file_path = output_dir / file_name

    logger.info(f"Starting weather table build...")
    df = build_aggregated_weather_dataset(root_path)
    logger.info(f"Final dataset shape: {df.shape}")

    df.to_csv(save_file_path, index=False)
    logger.info(f"Successfully saved to {save_file_path}")

if __name__ == "__main__":
    main()