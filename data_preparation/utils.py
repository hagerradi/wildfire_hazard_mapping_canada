import math
import os
from pathlib import Path

from matplotlib import pyplot as plt

from data_preparation.paths import OUTPUT_BURN_PROB_PATH

feature_count_map = {
    "ignition_prob": 1,
    "esc_fire_prob": 1,
    "weather_params": 16, # mean, var for 8 features
    "fuel_grid": 1,
    "wind_grid": 16,  # u, v for 8 directions
    "elevation_grid": 1,
    "out_burn_prob": 1,
}

def find_burn_prob_file(root_dir: str, hex_id: str) -> str:
    pattern = f"hex_{hex_id}_*_iter_bp.tif"
    matches = list(Path(os.path.join(root_dir, OUTPUT_BURN_PROB_PATH)).glob(pattern))

    if not matches:
        raise FileNotFoundError(f"No bp.tif file found matching pattern {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"Multiple bp files found: {matches}")

    return str(matches[0])

def find_hex_ids(root_dir:str)->list:
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
            im = ax.imshow(img_data, cmap='gray')  # noqa: F841
            ax.set_title(f"Window {i}")
        
        # Hide axis ticks for all subplots (cleaner look)
        ax.axis('off')

    plt.tight_layout()
    plt.show()