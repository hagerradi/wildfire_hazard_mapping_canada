import numpy as np

from data_preparation.spatial.utils import FUEL_GROUP_MAP

AVAILABLE_DATA_SOURCES = ["grid", "weather", "fire_size"]
MAX_FUEL_GRID = float(max(FUEL_GROUP_MAP.values()))


def get_data_source_class(name: str):
    """
    Returns the source class dynamically to avoid circular imports.
    """
    if name in ["grid"]:
        from src.datasets.sources import GridSource

        return GridSource
    elif name in ["weather", "fire_size"]:
        from src.datasets.sources import TabularSource

        return TabularSource
    else:
        raise ValueError(f"Unknown data source type: {name}. Available: {AVAILABLE_DATA_SOURCES}")


def get_data_source_param_class(name: str):
    """
    Returns the param class dynamically to avoid circular imports.
    """
    if name in ["grid"]:
        from src.config import GridParams

        return GridParams
    elif name in ["weather", "fire_size"]:
        from src.config import TabularParams

        return TabularParams
    else:
        raise ValueError(f"Unknown data source type: {name}. Available: {AVAILABLE_DATA_SOURCES}")


def get_dataset_dimensions(dataset) -> tuple[int | None, dict[str, int]]:
    """
    Extracts spatial and tabular dimensions from a MultiSourceDataset.
    """
    sources = getattr(dataset, "sources", {})

    spatial_channels = None
    auxiliary_input_dims = {}

    for name, source in sources.items():
        if name == "grid":
            spatial_channels = source.input_dim()
        else:
            auxiliary_input_dims[name] = source.input_dim()

    return spatial_channels, auxiliary_input_dims


def fill_nan_channel_mean_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Fills NaNs in a (H, W, C) array with the mean of the corresponding channel.
    Modifies the array in-place.
    """
    # 1. Calculate the mean of each channel, ignoring NaNs
    # axis=(0, 1) aggregates over Height and Width, leaving (C,)
    channel_means = np.nanmean(arr, axis=(0, 1))

    # 2. Find the indices where values are NaN
    # This returns a boolean mask of shape (H, W, C)
    nan_mask = np.isnan(arr)

    # 3. Replace NaNs
    # We grab the specific channel index (2) from the nan locations
    # and map them to the calculated means.
    arr[nan_mask] = np.take(channel_means, np.where(nan_mask)[2])

    return arr


def one_hot_encode(arr: np.ndarray, channel_idx: int, num_classes: int) -> np.ndarray:
    """
    Replaces the nth channel with its one-hot encoded version.
    Input: (H, W, C)
    Output: (H, W, C - 1 + num_classes)
    """
    # 1. Split the array
    # left: (H, W, n)
    left_part = arr[:, :, :channel_idx]

    # right: (H, W, C - n - 1)
    right_part = arr[:, :, channel_idx + 1 :]

    # target: (H, W)
    target_channel = arr[:, :, channel_idx]

    nan_mask = np.isnan(target_channel)

    # Replace NaN with 0 (or any safe index) temporarily so .astype(int) doesn't crash
    # We use np.nan_to_num to swap NaN -> 0 safely
    safe_target = np.nan_to_num(target_channel, nan=0).astype(int)

    # 3. One-Hot Encode using the "safe" integers
    encoded_part = np.eye(num_classes, dtype=arr.dtype)[safe_target]

    # 4. Zero out the vectors where the original value was NaN
    # Before this, the NaNs were encoded as Class 0 (because we filled with 0)
    # This step corrects that by setting them to [0, 0, 0...]
    encoded_part[nan_mask] = 0  # Nan is no fuel

    # 3. Concatenate along the channel axis (last axis)
    return np.concatenate([left_part, encoded_part.astype(np.float32), right_part], axis=-1)  # (H,W,C+14)


def log_norm(out_arr: np.ndarray, multiplier: int = 1000) -> np.ndarray:
    """
    Normalize the output burn prob array using log norm
    """
    return np.log1p(multiplier * out_arr) / np.log1p(multiplier)


def output_burn_prob_norm(
    output_arr: np.ndarray,
    burn_prob_max: float,
    burn_prob_min: float,
    out_norm: str,
    target_log_mean: float | None = None,
    target_log_std: float | None = None,
) -> np.ndarray:
    """
    Normalize the output target map.
    """
    if out_norm == "min_max":
        output_arr = (output_arr - burn_prob_min) / (burn_prob_max - burn_prob_min)
        output_arr = np.clip(output_arr, 0.0, 1.0)
    elif out_norm == "log":
        output_arr = log_norm(output_arr).astype(np.float32)
    elif out_norm == "log_standard":
        if target_log_mean is None or target_log_std is None:
            raise ValueError("target_log_mean and target_log_std are required for out_norm='log_standard'.")
        if target_log_std <= 0.0:
            raise ValueError(f"target_log_std must be positive for out_norm='log_standard', got {target_log_std}.")
        output_arr = (np.log1p(np.clip(output_arr, a_min=0.0, a_max=None)) - target_log_mean) / target_log_std
    elif out_norm in {"none", "total_iters", "season_cause_iters"}:
        pass
    else:
        raise ValueError(f"Unsupported output normalization: {out_norm!r}")
    return output_arr
