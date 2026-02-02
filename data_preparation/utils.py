import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Callable
from sklearn.model_selection import train_test_split

from data_preparation.paths import OUTPUT_BURN_PROB_PATH

feature_names = ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid", "out_grid"]
WEATHER_FEATURE_COLS = ["temp", "rh", "ws", "wd", "prec", "ffmc", "dmc", "dc", "isi", "bui", "fwi"]
HEX_ID_NA = ["52", "53", "04", "25", "47", "48"]

feature_count_map = {
    "ignition_prob": 1,
    "esc_fire_prob": 1,
    "weather_params": 16,  # mean, var for 8 features
    "fuel_grid": 1,
    "wind_grid": 16,  # u, v for 8 directions
    "elevation_grid": 1,
    "out_burn_prob": 1,
}

def load_all_data(root_dir: Path, pattern: str, load_function: Callable | None = None) -> pd.DataFrame:
    """
    Orchestrates the finding, loading, merging files of a certain pattern across all hex folders
    """
    # 1. Locate all files to load using specific pattern
    files = sorted(root_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {root_dir}...")
    print(f"Found {len(files)} files to process.")

    # 2. Load (with option to use feature specific loader function) and stack all dataframes
    data_frames = []
    for f in files:
        print(f"Loading file {f}")
        if load_function:
            df = load_function(f)
        else:
            df = pd.read_csv(f)
        if df is not None:
            data_frames.append(df)

    full_df = pd.concat(data_frames, ignore_index=True)
    print(f"Aggregated raw shape: {full_df.shape}")

    return full_df


def find_simulation_output_file(root_dir: str, hex_id: str, output_type: str, season: str = None, cause: str = None) -> str:
    """
    Helper to get the right raster file.

    Args:
        root_dir (str): The root dir.
        hex_id (str): The hexel ID.
        output_type (str): Either 'count' or 'prob'.
        season (str): The season.
        cause (str): The cause.

    """
    outputs_dir = os.path.join(root_dir, OUTPUT_BURN_PROB_PATH)

    if output_type == "count":
        suffix = "_bc.tif"
    elif output_type == "prob":
        suffix = "_bp.tif"
    else:
        raise ValueError(f"Invalid output_type: {output_type}.")

    if season is not None and cause is not None:
        fname = f"hex_{hex_id}_season_{season}_cause_{cause}{suffix}"
        return os.path.join(outputs_dir, fname)

    else:
        pattern = f"hex_{hex_id}_*iter{suffix}"
        matches = list(Path(outputs_dir).glob(pattern))

        if not matches:
            raise FileNotFoundError(f"No global file found matching '{pattern}' in {outputs_dir}")
        if len(matches) > 1:
            raise RuntimeError(f"Multiple {output_type} files found: {matches}")

        return str(matches[0])


def find_hex_ids(root_dir: str) -> list:
    hex_ids = []
    try:
        with os.scandir(root_dir) as entries:
            for entry in entries:
                # Check if it's a directory AND starts with 'hex'
                if entry.is_dir() and entry.name.startswith("hex"):
                    hex_ids.append(entry.name[3:])
    except FileNotFoundError:
        print(f"Directory not found: {root_dir}")
        return []

    return hex_ids


def get_min_max_hex_prob_df(data_dir: str) -> list:
    """create a list of burn prob dist for stratified sampling"""
    from data_preparation.grid_loader.output import load_output_burn_grid

    output_type, season, cause = "prob", None, None
    all_hex_ids = find_hex_ids(data_dir)
    hex_min_max_bp = []
    for hex_id in all_hex_ids:
        if hex_id in HEX_ID_NA:
            print("======Skipping hex=======", hex_id)
            continue
        root_dir = os.path.join(data_dir, f"hex{hex_id}")
        fpath = find_simulation_output_file(root_dir, hex_id, output_type, season=season, cause=cause)
        out_grid = load_output_burn_grid(fpath)
        out_grid_ravel = out_grid.ravel()
        area_burnt = len(out_grid_ravel[out_grid_ravel != 0.0]) / len(out_grid_ravel)
        hex_min_max_bp.append([hex_id, np.nanmin(out_grid), np.min(out_grid[out_grid != 0.0]), np.nanmax(out_grid), area_burnt])
    return hex_min_max_bp


def get_stratified_data_split(data_dir: str):
    """Stratified sampling for the valid data split"""
    hex_min_max_bp = get_min_max_hex_prob_df(data_dir)
    df_min_max = pd.DataFrame(hex_min_max_bp, columns=["hex_id", "min_prob", "min_except_0", "max_prob", "area_burnt"])
    df_min_max["area_bin"] = pd.qcut(df_min_max["area_burnt"], q=2, labels=["LowArea", "HighArea"])
    df_min_max["max_bin"] = pd.qcut(df_min_max["max_prob"], q=2, labels=["LowMax", "HighMax"])

    df_min_max["strat_key"] = df_min_max["area_bin"].astype(str) + "_" + df_min_max["max_bin"].astype(str)

    train_val, test = train_test_split(df_min_max, test_size=5, stratify=df_min_max["strat_key"], random_state=42)
    train, val = train_test_split(train_val, test_size=5, stratify=train_val["strat_key"], random_state=42)

    print(f"Total: {len(df_min_max)} | Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print(f"List of val ids {list(val["hex_id"])}")
    print(f"List of test ids {list(test["hex_id"])}")


def get_processed_hex_ids(folder_path: str) -> list:
    """
    Finds all hex_ids from the metadata df files
    """
    folder = Path(folder_path)
    hex_ids = []
    for file_path in folder.glob("meta_hex_*.csv"):
        filename_no_ext = file_path.stem
        extracted_id = filename_no_ext.removeprefix("meta_hex_")

        hex_ids.append(extracted_id)

    return hex_ids


def plot_split_window_hexel(windows, channel_index=0, max_cols=5, figsize=(15, 15)):
    """
    Plots a list/array of 3D windows in a subplot grid.

    Args:
        windows (list or np.ndarray): List of windows. Shape (N, H, W, C).
        channel_index (int): The channel to visualize (e.g., 0 for Red/Band1).
        max_cols (int): Maximum number of columns in the grid.
        figsize (tuple): Figure size (width, height).
    """
    num_windows = len(windows)

    if num_windows == 0:
        print("No windows to plot.")
        return

    # Calculate grid dimensions
    num_cols = min(num_windows, max_cols)
    num_rows = math.ceil(num_windows / num_cols)

    # Create subplots
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Flatten axes for easy iteration (handle case where axes is not a list)
    axes = [axes] if num_windows == 1 else axes.flatten()

    for i in range(len(axes)):
        ax = axes[i]

        if i < num_windows:
            window = windows[i]

            # Extract specific channel
            if window.ndim == 3:  # noqa: SIM108
                # Shape (H, W, C) -> Extract channel
                img_data = window[:, :, channel_index]
            else:
                # Fallback if window is already 2D
                img_data = window

            # Plot
            im = ax.imshow(img_data, cmap="gray")  # noqa: F841
            ax.set_title(f"Window {i}")

        # Hide axis ticks for all subplots (cleaner look)
        ax.axis("off")

    plt.tight_layout()
    plt.show()
