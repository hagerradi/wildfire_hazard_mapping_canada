import math

from matplotlib import pyplot as plt

feature_count_map = {
    "ignition_prob": 1,
    "esc_fire_prob": 1,
    "weather_params": 16, # mean, var for 8 features
    "fuel_grid": 1,
    "wind_grid": 16,  # u, v for 8 directions
    "elevation_grid": 1,
    "out_burn_prob": 1,
}

IGNITION_PROB_PATH =  "ignitions_module/ignition_grids"
FIRE_ZONE_GRID_PATH =  "mapped_inputs/cfrs.asc"
ESC_FIRE_DIST_PATH = "ignitions_module/Nb_ignitions_zone_season_cause_"

FUEL_GRID_PATH = "mapped_inputs/fbp.asc"
FUEL_TABLE_PATH = "mapped_inputs/Fuel_table.lut"

ELEVATION_GRID_PATH = "mapped_inputs/elev.asc"

WEATHER_LIST_PATH = "burning_conditions_module"
WIND_GRID_DIR_PATH = "burning_conditions_module/wind_grids"

OUTPUT_BURN_PROB_PATH = "outputs"


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