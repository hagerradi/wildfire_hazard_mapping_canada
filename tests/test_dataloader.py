import json
import os
import shutil
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
import torchvision.transforms.functional as F

from src.datasets.dataloader import GridDataset
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

        # Create dummy CSV
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
            arr[1, 1, :] = np.nan
            arr[10, 20, :] = np.nan
            np.save(os.path.join(tmpdir, fname), arr)

        yield tmpdir, train_csv, val_csv, test_csv
    finally:
        shutil.rmtree(tmpdir)


def test_len_and_getitem(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir
    ds = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=None,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )
    assert len(ds) == 3
    x, y, mask = ds[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert x.shape[1:] == y.shape[1:] == mask.shape[1:]
    assert mask.dtype == torch.bool


def test_one_hot_encoding(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir
    ds = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="min_max",
        fuel_feats_encoding="one_hot",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=None,
        feature_names_list=["fuel_grid"],
    )
    x, y, mask = ds[0]
    # Should have more channels due to one-hot
    assert x.shape[0] > 7


def test_feature_names_list(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir
    ds = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=None,
        feature_names_list=["fuel_grid"],
    )
    x, y, mask = ds[0]
    # Only selected features
    assert x.shape[0] == 1


def test_mask_threshold(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir
    # Set threshold above 1.0 so no samples are valid
    ds = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="min_max",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=1.0,
        transform=None,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )
    assert len(ds) == 0


def test_output_normalization_iters(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir
    ds = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=None,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )
    _, y, _ = ds[0]
    # Output should be normalized by total_unique_iters (10)
    arr = np.load(os.path.join(tmpdir, "sample_0.npy")).astype(np.float32)
    expected = arr[:, :, -1] / 10
    # Mask out nan locations for comparison
    y_np = y.squeeze().numpy()
    mask = ~np.isnan(expected)
    np.testing.assert_allclose(y_np[mask], expected[mask], rtol=1e-5, atol=1e-5)


def test_transforms(temp_data_dir):
    tmpdir, train_csv, _, _ = temp_data_dir

    # original data
    ds_orig = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=None,  # no flipping
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )
    x_orig, y_orig, _ = ds_orig[0]

    # mock random flip config and augmented data
    class MockConfig:
        transforms_list = ["random_flip"]
        augmentation_prob = 1.0

    transform_flip = setup_augmentations(MockConfig())

    ds_flip = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=transform_flip,  # apply flipping
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )

    x_flip, _, _ = ds_flip[0]

    # possible augmented versions
    possible_h = F.hflip(x_orig)
    possible_v = F.vflip(x_orig)

    # check if augmented tensor is one of the flips
    assert torch.equal(x_flip, possible_h) or torch.equal(x_flip, possible_v)

    # mock random rotate and augmented data
    class MockConfigRot:
        transforms_list = ["random_rotate"]
        augmentation_prob = 1.0

    transform_rot = setup_augmentations(MockConfigRot())

    ds_rot = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=transform_rot,  # apply rotation
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )

    # here we test with target since transform should apply to it as well
    _, y_rot, _ = ds_rot[0]

    # possible rotations expected
    is_90 = torch.equal(y_rot, torch.rot90(y_orig, 1, dims=[1, 2]))
    is_180 = torch.equal(y_rot, torch.rot90(y_orig, 2, dims=[1, 2]))
    is_270 = torch.equal(y_rot, torch.rot90(y_orig, 3, dims=[1, 2]))

    assert is_90 or is_180 or is_270


def test_split_transforms_config(temp_data_dir):
    tmpdir, train_csv, val_csv, test_csv = temp_data_dir

    # mock transforms config.
    class MockConfig:
        transforms_list = ["random_flip", "random_rotate"]
        augmentation_prob = 0.5

    train_transform = setup_augmentations(MockConfig())

    ds_train = GridDataset(
        csv_name=train_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        transform=train_transform,  # use transforms for train
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )

    # no passing of transforms for val.
    ds_val = GridDataset(
        csv_name=val_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )

    # no passing of transforms for test
    ds_test = GridDataset(
        csv_name=test_csv,
        root_dir=tmpdir,
        filename_col="filename",
        out_norm="total_iters",
        fuel_feats_encoding="ordinal",
        normalize_fuel_feats_ordinal=True,
        modelling_approach="2",
        valid_mask_threshold=0.0,
        feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"],
    )

    # assert that train has a transforms
    assert ds_train.transform is not None
    assert callable(ds_train.transform)

    # assert that val. and test do not have any
    assert ds_val.transform is None
    assert ds_test.transform is None
