import json
import os
from collections.abc import Callable
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from config import DataConfig
from data_preparation.grid_loader.utils import BURN_COUNT_MAX, BURN_COUNT_MIN, fuel_ranking, get_range_burn_prob
from src.datasets.transforms import setup_augmentations
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_burn_prob_norm
from utils import seed_worker

MAX_FUEL_GRID = float(max(fuel_ranking.values()))
MIN_FUEL_GRID = float(min(fuel_ranking.values()))

class GridDataset(Dataset):
    """
    Base Spatial Dataset.
    Responsibility: Loads, normalizes, and returns spatial grids (Image, Mask, Target).
    """

    def __init__(
        self,
        csv_name: str,
        root_dir: str,
        feature_names_list: list[str],
        filename_col: str = "filename",
        out_norm: str = "min_max",
        fuel_feats_encoding: str = "ordinal",
        normalize_fuel_feats_ordinal: bool | None = True,
        modelling_approach: str = "2",
        valid_mask_threshold: float = 0.01,
        transform: Callable | None = None,
    ):
        """
        Args:
            csv_name (str): Path to the csv file with annotations.
            root_dir (str): Directory with all the .npy files.
            filename_col (str): Column name in CSV containing the filenames.
            out_norm (str): How to normalize the output burn counts for modelling approach 2. [Options: total_iters, season_cause_iters, min_max]
            fuel_feats_encoding(str): How to process the fuel features [Options: ordinal, one_hot]
            normalize_fuel_feats_ordinal (bool): If we want to normalize the ordinal encoded fuel feats
            modelling_approach (str): The approach used for modelling
            mask_threshold (float): The threshold for how much valid data should be present in a data sample
            transform (callable, optional): Optional transform to be applied on a sample.
            feature_names_list (list): List of features being used for training ((options: None or feature list) All feats: ["ignition_grid", "esc_fires_grid", "fuel_grid", "elevation_grid", "weather_grid", "wind_grid"])
        """
        self.filename_col = filename_col
        self.transform = transform
        self.out_norm = out_norm
        self.modelling_approach = modelling_approach
        self.root_dir = root_dir
        self.valid_mask_threshold = valid_mask_threshold
        self.feature_names_list = feature_names_list
        self.fuel_feats_encoding = fuel_feats_encoding
        self.normalize_fuel_feats_ordinal = normalize_fuel_feats_ordinal

        if not self.feature_names_list:
            raise ValueError(
                "Feature names list should never be empty or None. Valid list: [ignition_grid, esc_fires_grid, fuel_grid, elevation_grid, weather_grid, wind_grid]"
            )

        # Load Metadata
        self.metadata_df = pd.read_csv(os.path.join(self.root_dir, csv_name))
        self.metadata_df = self.metadata_df[self.metadata_df["valid_ratio"] > self.valid_mask_threshold]
        self.all_files = list(self.metadata_df[filename_col])

        # Setup Normalization Constants
        self.BURN_PROB_MAX, self.BURN_PROB_MIN = 1.0, 0.0
        if self.modelling_approach == "1" and self.out_norm == "min_max":
            self.BURN_PROB_MAX, self.BURN_PROB_MIN = get_range_burn_prob(os.path.dirname(self.root_dir))

        if self.out_norm == "total_iters":
            self.out_norm_array = list(
                self.metadata_df["total_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for all seasons and causes
        elif self.out_norm == "season_cause_iters":
            self.out_norm_array = list(
                self.metadata_df["season_cause_unique_iters"]
            )  # For modelling approach 2: total number of unique interations that produced fires for a single season and cause

        # Load Channel Maps
        with open(os.path.join(self.root_dir, f"feature_channel_map_{self.modelling_approach}.json")) as f:
            self.channel_feature_map = json.load(f)
            
            # We store this as 'base_channel_indices' because one-hot encoding might expand it dynamically per item
            self.base_channel_indices = [
                item for key in self.feature_names_list for item in self.channel_feature_map[key]
            ]
            
            self.fuel_feat_index = -1
            if "fuel_grid" in self.feature_names_list:
                self.fuel_feat_index = self.channel_feature_map["fuel_grid"][0]

    def __len__(self):
        return len(self.metadata_df)

    def _process_spatial_data(self, data, idx):
        """
        Shared logic for processing the spatial tensor.
        Extracts inputs/outputs, handles NaNs, One-Hot Encoding, and Normalization.
        """
        # 1. Split Input/Output
        input_arr = data[:, :, :-1]
        output_arr = data[:, :, -1]
        output_arr[np.isnan(output_arr)] = 0.0

        # 2. Generate Mask
        assert np.all(np.isnan(input_arr) == np.isnan(input_arr[..., :1])), "NaN mask differs across channels!"
        mask = ~np.isnan(input_arr[:, :, 0])

        # 3. Handle Feature Encoding
        # We use a local variable `current_indices` so we don't modify self.base_channel_indices
        current_indices = self.base_channel_indices[:]
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":  # (H,W,C+20)
            num_classes = int(MAX_FUEL_GRID + 1)
            # This returns a LARGER array with new channels appended
            input_arr = one_hot_encode(arr=input_arr, channel_idx=self.fuel_feat_index, num_classes=num_classes)
            
            # Recalculate indices for this specific item
            old_fuel_idx = self.fuel_feat_index
            current_indices = [i for i in current_indices if i != old_fuel_idx]
            
            # Add the new one-hot indices (which are at the end of the array)
            current_indices = (
                current_indices[:old_fuel_idx]
                + list(range(old_fuel_idx, old_fuel_idx + num_classes))
                + [i + num_classes - 1 for i in current_indices[old_fuel_idx:]]
            )

        # 4. Fill NaNs
        input_arr = fill_nan_channel_mean_numpy(input_arr)

        # 5. Ordinal Normalization
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "ordinal":
            input_arr[:, :, self.fuel_feat_index][~mask] = 0.0
            if self.normalize_fuel_feats_ordinal:
                input_arr[:, :, self.fuel_feat_index] = (input_arr[:, :, self.fuel_feat_index] - MIN_FUEL_GRID) / (
                    MAX_FUEL_GRID - MIN_FUEL_GRID
                )

        # 6. Channel Selection
        if current_indices:
            input_arr = input_arr[:, :, current_indices]

        # 7. Output Normalization
        if self.modelling_approach == "2":
            if self.out_norm == "min_max":
                output_arr = (output_arr - BURN_COUNT_MIN) / (BURN_COUNT_MAX - BURN_COUNT_MIN)
            elif self.out_norm in ["total_iters", "season_cause_iters"]:
                output_arr /= self.out_norm_array[idx]
        elif self.modelling_approach == "1":
            output_arr = output_burn_prob_norm(
                output_arr=output_arr, burn_prob_max=self.BURN_PROB_MAX, burn_prob_min=self.BURN_PROB_MIN, out_norm=self.out_norm
            )

        # 8. To Tensor
        input_tensor = torch.from_numpy(input_arr).permute(2, 0, 1)
        output_tensor = torch.from_numpy(np.expand_dims(output_arr, 0))
        mask_tensor = torch.from_numpy(np.expand_dims(mask, 0))

        return input_tensor, output_tensor, mask_tensor # (C, H, W), (1, H, W), (1, H, W)

    def __getitem__(self, idx):
        filename = self.all_files[idx]
        file_path = os.path.join(self.root_dir, filename)
        data = np.load(file_path).astype(np.float32)

        input_tensor, output_tensor, mask_tensor = self._process_spatial_data(data, idx)

        if self.transform:
            input_tensor, output_tensor, mask_tensor = self.transform(input_tensor, output_tensor, mask_tensor)

        return input_tensor, output_tensor, mask_tensor

    @property
    def num_grid_channels(self) -> int:
        """
        Returns the final number of channels the grids
        Automatically accounts for:
        1. Channels dropped by subclasses (e.g. weather grid)
        2. Channels added by One-Hot Encoding
        """
        # 1. Start with the raw indices (subclasses like WeatherGridDataset may have filtered this list already)
        count = len(self.base_channel_indices)
        
        # 2. Adjust for One-Hot Encoding if active
        # Logic: Drop the 1 ordinal channel and add N one-hot channels
        if "fuel_grid" in self.feature_names_list and self.fuel_feats_encoding == "one_hot":
            num_classes = int(MAX_FUEL_GRID + 1)
            count = count - 1 + num_classes
            
        return count

class WeatherGridDataset(GridDataset):
    """
    Extends GridDataset to handle Tabular Weather lookup.
    Responsibility: 
    1. Intercepts data loading to extract Weather Zone ID.
    2. Removes Weather Zone channel from Spatial Tensor.
    3. Samples Weather sequence.
    Returns: (Image, Mask, Target, Weather)
    """
    def __init__(
        self,
        weather_table_path: str,
        weather_samples_per_item: int = 128,
        weather_channel_name: str = "weather_grid",
        weather_features: list[str] = ['temp', 'rh', 'prec', 'ffmc', 'dmc', 'dc', 'isi', 'bui'],
        **kwargs 
    ):
        # Initialize Parent (loads metadata, basic maps)
        super().__init__(**kwargs)
        self.weather_table_path = weather_table_path
        self.weather_samples_per_item = weather_samples_per_item
        self.weather_features = weather_features
        self.weather_channel_name = weather_channel_name

        # Validate Weather Requirements
        if self.weather_channel_name not in self.channel_feature_map:
            raise ValueError(f"Weather channel '{self.weather_channel_name}' not found in feature map.")
        
        # Identify the Weathe Zone ID index
        self.weather_zone_raw_idx = self.channel_feature_map[self.weather_channel_name][0]

        #  Modify Parent State, remove the weather channel from the spatial features list, _process_spatial_data drops it
        zone_idx = self.channel_feature_map[self.weather_channel_name]
        self.base_channel_indices = [i for i in self.base_channel_indices if i not in zone_idx]

        #  Build Lookup Table
        print(f"Loading weather table from {self.weather_table_path}...")
        df_weather = pd.read_csv(self.weather_table_path)
        missing_cols = [c for c in self.weather_features if c not in df_weather]
        if missing_cols:
            raise ValueError(f"Missing weather features: {missing_cols}")
        self.weather_lut = {}
        for zone, group in df_weather.groupby('wx_zone'):
            feats = group[self.weather_features].values.astype(np.float32)
            self.weather_lut[int(zone)] = feats
        print(f"Weather LUT built for {len(self.weather_lut)} zones.")

    def __getitem__(self, idx):
        filename = self.all_files[idx]
        file_path = os.path.join(self.root_dir, filename)
        
        # 1. Load Raw Data
        data = np.load(file_path).astype(np.float32)

        # 2. Process Spatial
        input_tensor, output_tensor, mask_tensor = self._process_spatial_data(data, idx)

        # 3. Sample Weather using Mode Logic
        num_feats = len(self.weather_features)
        spatial_mask = mask_tensor[0].numpy().astype(bool)  # Get the boolean mask from the tensor (Mask is shape 1,H,W -> need H,W)
        raw_zone_channel = data[:, :, self.weather_zone_raw_idx].copy() # Use the boolean mask to filter valid zones
        valid_zones = raw_zone_channel[spatial_mask]
        if valid_zones.size > 0:
            values, counts = np.unique(valid_zones, return_counts=True)
            mode_zone = int(values[np.argmax(counts)])
            if mode_zone in self.weather_lut:
                candidates = self.weather_lut[mode_zone]
                # If we have enough data, sample; otherwise pad
                if len(candidates) >= self.weather_samples_per_item:
                    sample_indices = np.random.choice(len(candidates),size=self.weather_samples_per_item,replace=False)
                    weather_samples_np = candidates[sample_indices]
                else:
                    # Random sampling with replacement if short, or just repeat/pad?
                    # The original code had logic: replace=(len < size)
                    sample_indices = np.random.choice(len(candidates),size=self.weather_samples_per_item,replace=True)
                    weather_samples_np = candidates[sample_indices]
            else:
                # Mode zone not in LUT
                weather_samples_np = np.zeros((self.weather_samples_per_item, num_feats), dtype=np.float32)
        else:
            # No valid zones found in mask
            weather_samples_np = np.zeros((self.weather_samples_per_item, num_feats), dtype=np.float32)
        weather_tensor = torch.from_numpy(weather_samples_np)

        # 5. Apply Transforms (Spatial Only)
        if self.transform:
            input_tensor, output_tensor, mask_tensor = self.transform(input_tensor, output_tensor, mask_tensor)
        
        return input_tensor, output_tensor, mask_tensor, weather_tensor # (C, H, W), (1, H, W), (1, H, W), (N, F), where N=num of subsamples and F=dimension of weather

    @property
    def num_weather_features(self) -> int:
        """Returns the number of tabular weather features."""
        return len(self.weather_features)

def get_train_val_dataloader(config: DataConfig, modelling_approach: str = "1", seed: int = 42):
    """
    Creates and returns a DataLoader with deterministic shuffling
    """
    root_dir = config.root_dir
    train_csv_name = config.train_split
    val_csv_name = config.val_split
    filename_col = config.filename_col
    batch_size = config.batch_size
    num_workers = config.num_workers
    out_norm = config.output_normalization
    transform = setup_augmentations(config)
    feature_names_list = config.feature_names_list
    fuel_feats_encoding = config.fuel_feats_encoding
    normalize_fuel_feats_ordinal = config.normalize_fuel_feats_ordinal
    valid_mask_threshold = config.valid_mask_threshold
    weather_table_path = config.weather_table_path
    weather_samples_per_item = config.weather_samples_per_item
    weather_channel_name = config.weather_channel_name
    weather_features = config.weather_features

    if config.weather_table_path is None:
        train_dataset = GridDataset(
            csv_name=train_csv_name,
            root_dir=root_dir,
            feature_names_list=feature_names_list,
            filename_col=filename_col,
            out_norm=out_norm,
            fuel_feats_encoding=fuel_feats_encoding,
            normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
            modelling_approach=modelling_approach,
            valid_mask_threshold=valid_mask_threshold,
            transform=transform,
        )
        val_dataset = GridDataset(
            csv_name=val_csv_name,
            root_dir=root_dir,
            feature_names_list=feature_names_list,
            filename_col=filename_col,
            out_norm=out_norm,
            fuel_feats_encoding=fuel_feats_encoding,
            normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
            modelling_approach=modelling_approach,
            valid_mask_threshold=valid_mask_threshold,
            transform=None,  # no transforms for val. set
        )
    else:
        train_dataset = WeatherGridDataset(
            csv_name=train_csv_name,
            root_dir=root_dir,
            feature_names_list=feature_names_list,
            filename_col=filename_col,
            out_norm=out_norm,
            fuel_feats_encoding=fuel_feats_encoding,
            normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
            modelling_approach=modelling_approach,
            valid_mask_threshold=valid_mask_threshold,
            transform=transform,
            weather_table_path=weather_table_path,
            weather_samples_per_item=weather_samples_per_item,
            weather_channel_name=weather_channel_name,
            weather_features=weather_features
        )
        val_dataset = WeatherGridDataset(
            csv_name=val_csv_name,
            root_dir=root_dir,
            feature_names_list=feature_names_list,
            filename_col=filename_col,
            out_norm=out_norm,
            fuel_feats_encoding=fuel_feats_encoding,
            normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
            modelling_approach=modelling_approach,
            valid_mask_threshold=valid_mask_threshold,
            transform=transform,
            weather_table_path=weather_table_path,
            weather_samples_per_item=weather_samples_per_item,
            weather_channel_name=weather_channel_name,
            weather_features=weather_features
        )

    # Generators & Loaders
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        worker_init_fn=seed_worker,
        generator=g,
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )

    return train_loader, val_loader


def get_test_loader(config: DataConfig, modelling_approach: str = "1", seed: int = 42):
    """
    Creates and returns the test loader
    """
    root_dir = config.root_dir
    test_csv_name = config.test_split
    filename_col = config.filename_col
    batch_size = config.batch_size
    num_workers = config.num_workers
    out_norm = config.output_normalization
    feature_names_list = config.feature_names_list
    fuel_feats_encoding = config.fuel_feats_encoding
    normalize_fuel_feats_ordinal = config.normalize_fuel_feats_ordinal
    valid_mask_threshold = config.valid_mask_threshold

    test_dataset = GridDataset(
        csv_name=test_csv_name,
        root_dir=root_dir,
        feature_names_list=feature_names_list,
        filename_col=filename_col,
        out_norm=out_norm,
        fuel_feats_encoding=fuel_feats_encoding,
        normalize_fuel_feats_ordinal=normalize_fuel_feats_ordinal,
        modelling_approach=modelling_approach,
        valid_mask_threshold=valid_mask_threshold,
        transform=None,  # no transforms for test set
    )

    g = torch.Generator()
    g.manual_seed(seed)

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, worker_init_fn=seed_worker, generator=g
    )

    return test_loader
