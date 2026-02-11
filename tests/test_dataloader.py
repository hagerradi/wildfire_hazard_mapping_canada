import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as F

from src.config import DataSourceConfig, GridParams
from src.datasets.dataset import MultiSourceDataset
from src.datasets.sources import GridSource, WeatherSource
from src.datasets.transforms import setup_augmentations


@pytest.fixture
def temp_data_dir():
    tmpdir = tempfile.mkdtemp()
    try:
        # Create dummy feature_channel_map
        feature_channel_map = {
            "ignition_grid": [0],
            "esc_fires_grid": [1],
            "fuel_grid": [2],
            "elevation_grid": [3],
            "weather_grid": [4],
            "wind_grid": [5],
        }
        with open(os.path.join(tmpdir, "feature_channel_map_2.json"), "w") as f:
            json.dump(feature_channel_map, f)

        # Create dummy metadata CSV
        filenames = []
        valid_ratios = []
        total_unique_iters = []
        season_cause_unique_iters = []
        for i in range(3):
            fname = f"sample_{i}.npy"
            filenames.append(fname)
            valid_ratios.append(1.0)
            total_unique_iters.append(10)
            season_cause_unique_iters.append(5)
        df = pd.DataFrame(
            {
                "filename": filenames,
                "valid_ratio": valid_ratios,
                "total_unique_iters": total_unique_iters,
                "season_cause_unique_iters": season_cause_unique_iters,
            }
        )
        train_csv = "train.csv"
        df.to_csv(os.path.join(tmpdir, train_csv), index=False)
        val_csv = "val.csv"
        df.to_csv(os.path.join(tmpdir, val_csv), index=False)
        test_csv = "test.csv"
        df.to_csv(os.path.join(tmpdir, test_csv), index=False)

        # Create dummy npy files
        for fname in filenames:
            arr = np.random.rand(32, 32, 36).astype(np.float32)
            # Add some NaNs to input channels
            arr[:, :, 4] = 100.0  # Force this channel to be '100.0' so it matches weather CSV below.
            arr[1, 1, :] = np.nan
            arr[10, 20, :] = np.nan
            np.save(os.path.join(tmpdir, fname), arr)

        # Create dummy weather table csv
        weather_feats = ["temp", "rh", "prec", "ffmc", "dmc", "dc", "isi", "bui"]
        data = {feat: np.random.rand(5) for feat in weather_feats}
        data["wx_zone"] = [100, 100, 100, 200, 200]  # 3 samples for zone 100
        weather_df = pd.DataFrame(data)
        weather_csv = "weather_table.csv"
        weather_df.to_csv(os.path.join(tmpdir, weather_csv), index=False)

        yield tmpdir, train_csv, val_csv, test_csv, weather_csv, weather_feats

    finally:
        shutil.rmtree(tmpdir)


def test_multi_source_integration(temp_data_dir):
    tmpdir, train_csv, val_csv, test_csv, weather_csv, weather_feats = temp_data_dir
    grid_source = GridSource(
        root_dir=tmpdir,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        modelling_approach="2",
        fuel_feats_encoding="ordinal",
    )
    weather_source = WeatherSource(
        weather_samples_csv_name=weather_csv,
        root_dir=tmpdir,
        feature_names_list=weather_feats,
        modelling_approach="2",
        num_samples_per_patch=2,
    )
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source, "weather": weather_source})
    sample = ds[0]

    assert "grid" in sample.keys()
    assert "weather" in sample.keys()
    input_arr, target, mask = sample["grid"]
    assert isinstance(input_arr, torch.Tensor)
    assert input_arr.shape[0] == 3
    weather = sample["weather"]
    assert weather.shape == (2, len(weather_feats))


def test_grid_one_hot_encoding(temp_data_dir):
    tmpdir, train_csv, _, _, _, _ = temp_data_dir
    grid_source = GridSource(
        root_dir=tmpdir,
        feature_names_list=["fuel_grid"],
        modelling_approach="2",
        fuel_feats_encoding="one_hot",
        normalize_fuel_feats_ordinal=True,
    )
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source})
    sample = ds[0]
    x, y, mask = sample["grid"]
    # Should have more channels due to one-hot
    assert x.shape[0] > 1


def test_grid_feature_names_list(temp_data_dir):
    tmpdir, train_csv, _, _, _, _ = temp_data_dir
    grid_source = GridSource(root_dir=tmpdir, feature_names_list=["fuel_grid"], modelling_approach="2", fuel_feats_encoding="ordinal")
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source})

    sample = ds[0]
    x, y, mask = sample["grid"]

    # Assert shape is exactly 1
    assert x.shape[0] == 1


def test_mask_threshold(temp_data_dir):
    tmpdir, train_csv, _, _, _, _ = temp_data_dir
    # Set threshold above 1.0 so no samples are valid
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, valid_mask_threshold=1.0)

    assert len(ds) == 0


def test_grid_output_normalization_iters(temp_data_dir):
    tmpdir, train_csv, _, _, _, _ = temp_data_dir
    grid_source = GridSource(
        root_dir=tmpdir,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
        modelling_approach="2",
        out_norm="total_iters",
    )
    ds = MultiSourceDataset(csv_name="train.csv", root_dir=tmpdir, sources={"grid": grid_source})
    sample = ds[0]
    _, y, _ = sample["grid"]
    # Output should be normalized by total_unique_iters (10)
    arr = np.load(os.path.join(tmpdir, "sample_0.npy")).astype(np.float32)
    expected = arr[:, :, -1] / 10
    # Mask out nan locations for comparison
    y_np = y.squeeze().numpy()
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(y_np[mask], expected[mask], rtol=1e-5, atol=1e-5)


def test_grid_transforms(temp_data_dir):
    tmpdir, train_csv, _, _, _, _ = temp_data_dir
    # Get Original Data (No Transforms)
    ds_orig = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                modelling_approach="2",
                out_norm="total_iters",
                fuel_feats_encoding="ordinal",
                normalize_fuel_feats_ordinal=True,
                transform=None,  # No transforms here
            )
        },
    )
    x_orig, y_orig, _ = ds_orig[0]["grid"]

    # =============================================
    # Test 1: Mock Config for Augmentation (Flip)
    # =============================================
    flip_config = DataSourceConfig(
        name="grid",
        params=GridParams(
            feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"], transforms_list=["random_flip"], augmentation_prob=1.0
        ),
    )
    transform_flip = setup_augmentations(flip_config)
    ds_flip = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                modelling_approach="2",
                out_norm="total_iters",
                fuel_feats_encoding="ordinal",
                normalize_fuel_feats_ordinal=True,
                transform=transform_flip,  # Apply Flip Transform
            )
        },
    )
    x_flip, _, _ = ds_flip[0]["grid"]
    # possible augmented versions
    possible_h = F.hflip(x_orig)
    possible_v = F.vflip(x_orig)
    # check if augmented tensor is one of the flips
    assert torch.equal(x_flip, possible_h) or torch.equal(x_flip, possible_v)

    # =============================================
    # Test 2: Mock Config for Augmentation (Rotate)
    # =============================================
    rot_config = DataSourceConfig(
        name="grid",
        params=GridParams(
            feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"], transforms_list=["random_rotate"], augmentation_prob=1.0
        ),
    )
    transform_rot = setup_augmentations(rot_config)
    ds_rot = MultiSourceDataset(
        csv_name="train.csv",
        root_dir=tmpdir,
        sources={
            "grid": GridSource(
                root_dir=tmpdir,
                feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
                modelling_approach="2",
                out_norm="total_iters",
                fuel_feats_encoding="ordinal",
                normalize_fuel_feats_ordinal=True,
                transform=transform_rot,  # Apply Rotate Transform
            )
        },
    )
    # here we test with target since transform should apply to it as well
    _, y_rot, _ = ds_rot[0]["grid"]

    # possible rotations expected
    is_90 = torch.equal(y_rot, torch.rot90(y_orig, 1, dims=[1, 2]))
    is_180 = torch.equal(y_rot, torch.rot90(y_orig, 2, dims=[1, 2]))
    is_270 = torch.equal(y_rot, torch.rot90(y_orig, 3, dims=[1, 2]))

    assert is_90 or is_180 or is_270
