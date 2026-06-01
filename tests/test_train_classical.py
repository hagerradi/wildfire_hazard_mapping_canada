import json
from argparse import Namespace

import numpy as np
import pandas as pd

from src.config import Config
from src.train_classical import (
    ClassicalFeatureBuilder,
    compute_patch_metrics,
    extract_split_rows,
    predict_split,
    train_lightgbm,
)
from src.utils import AVAILABLE_METRICS


def _write_classical_fixture(tmp_path):
    root = tmp_path / "data"
    npy_dir = root / "numpy_files"
    npy_dir.mkdir(parents=True)
    channel_map = {
        "fuel_grid": [0],
        "elevation_grid": [1],
        "ignition_grid": [2],
        "firezones_grid": [3],
        "bp_out_grid": [4],
        "fi_out_grid": [5],
        "ros_out_grid": [6],
    }
    (root / "feature_channel_map_1.json").write_text(json.dumps(channel_map))

    records = []
    for idx, split in enumerate(["train", "train", "val", "test"]):
        arr = np.zeros((4, 4, 7), dtype=np.float32)
        arr[:, :, 0] = np.array(
            [
                [1, 1, 2, 2],
                [1, 0, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 0],
            ],
            dtype=np.float32,
        )
        arr[:, :, 1] = np.arange(16, dtype=np.float32).reshape(4, 4) + idx
        arr[:, :, 2] = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
        arr[:, :, 3] = 10
        arr[:, :, 4] = np.clip(arr[:, :, 2] * 0.1 + idx * 0.01, 0.0, 1.0)
        arr[:, :, 5] = arr[:, :, 2] * 100.0
        arr[:, :, 6] = arr[:, :, 2]
        path = npy_dir / f"patch_{idx}.npy"
        np.save(path, arr)
        records.append(
            {
                "filename": f"numpy_files/patch_{idx}.npy",
                "season": "all",
                "cause": "all",
                "hex_id": idx + 1,
                "window_id": idx,
                "row": idx * 4,
                "col": idx * 2,
                "valid_ratio": 1.0,
                "split": split,
            }
        )

    records_df = pd.DataFrame(records)
    for split in ["train", "val", "test"]:
        records_df[records_df["split"] == split].drop(columns=["split"]).to_csv(root / f"{split}_indices.csv", index=False)

    pd.DataFrame(
        {
            "WeatherZone": [10, 10],
            "Temperature": [20.0, 25.0],
            "RelativeHumidity": [30.0, 35.0],
        }
    ).to_csv(root / "weather_table_processed.csv", index=False)
    pd.DataFrame({"GRIDCODE": [10, 10], "NORM_LOG_SIZE_HA": [0.2, 0.8]}).to_csv(root / "df_fire_fru_processed.csv", index=False)
    return root


def _make_config(root) -> Config:
    return Config.model_validate(
        {
            "save_dir": str(root / "experiments"),
            "modelling_approach": "1",
            "seed": 42,
            "deterministic": True,
            "model": {"num_classes": 1, "input_branches": ["spatial", "auxiliary"]},
            "optimizer": {"name": "AdamW", "lr": 1e-3, "loss": "kl"},
            "training": {"max_epochs": 1, "log_every_n_epoch": 1},
            "evaluation": {"best_ckpt_metrics": ["ccc"], "best_ckpt_metrics_mode": ["max"]},
            "data": {
                "root_dir": str(root),
                "raw_data_dir": str(root),
                "train_split": "train_indices.csv",
                "val_split": "val_indices.csv",
                "test_split": "test_indices.csv",
                "batch_size": 2,
                "num_workers": 0,
                "valid_mask_threshold": 0.01,
                "input_sources": [
                    {
                        "name": "grid",
                        "params": {
                            "feature_names_list": ["ignition_grid", "fuel_grid", "elevation_grid"],
                            "target_name": "bp",
                            "out_norm": "min_max",
                            "fuel_feats_encoding": "one_hot",
                        },
                    },
                    {
                        "name": "tabular_weather",
                        "params": {
                            "csv_name": "weather_table_processed.csv",
                            "feature_names_list": ["Temperature", "RelativeHumidity"],
                            "fire_weather_zone_id_col": "WeatherZone",
                            "fire_weather_zone_selection_approach": "mode",
                        },
                    },
                    {
                        "name": "tabular_fire_size",
                        "params": {
                            "csv_name": "df_fire_fru_processed.csv",
                            "feature_names_list": ["NORM_LOG_SIZE_HA"],
                            "fire_weather_zone_id_col": "GRIDCODE",
                            "fire_weather_zone_selection_approach": "mode",
                        },
                    },
                ],
            },
            "logger": {
                "enabled": False,
                "project_name": "test",
                "workspace": "test",
                "experiment_name": "test",
            },
            "metrics": ["mae", "ccc", "iou_top10"],
            "data_prep": {"win_h": 4, "win_w": 4, "overlap_ratio": 0.0},
        }
    )


def test_classical_feature_builder_extracts_finite_features(tmp_path, monkeypatch):
    root = _write_classical_fixture(tmp_path)
    config = _make_config(root)
    monkeypatch.setattr("src.train_classical.get_range_elevation", lambda _root: (16.0, 0.0))
    monkeypatch.setattr("src.train_classical.get_range_output", lambda _root, _output_type: (1.0, 0.0))

    builder = ClassicalFeatureBuilder(config=config, local_windows=(3,))
    data = np.load(root / "numpy_files/patch_0.npy")
    features, target, mask = builder.patch_features(data, {"row": 0, "col": 0, "valid_ratio": 1.0})

    assert features.shape == (4, 4, len(builder.feature_names))
    assert target.shape == (4, 4)
    assert mask.all()
    assert np.isfinite(features).all()
    assert "weather_Temperature_mean" in builder.feature_names
    assert "fire_size_NORM_LOG_SIZE_HA_p90" in builder.feature_names


def test_classical_lightgbm_smoke_runs(tmp_path, monkeypatch):
    root = _write_classical_fixture(tmp_path)
    config = _make_config(root)
    monkeypatch.setattr("src.train_classical.get_range_elevation", lambda _root: (16.0, 0.0))
    monkeypatch.setattr("src.train_classical.get_range_output", lambda _root, _output_type: (1.0, 0.0))

    builder = ClassicalFeatureBuilder(config=config, local_windows=(3,))
    rng = np.random.default_rng(42)
    train_arrays = extract_split_rows(config, builder, "train_indices.csv", 8, rng, 0.5, 0.9)
    val_arrays = extract_split_rows(config, builder, "val_indices.csv", 8, rng, 0.5, 0.9)
    args = Namespace(
        objective="regression",
        n_estimators=3,
        learning_rate=0.1,
        num_leaves=7,
        min_child_samples=1,
        subsample=1.0,
        colsample_bytree=1.0,
        reg_alpha=0.0,
        reg_lambda=0.0,
        num_threads=1,
        early_stopping_rounds=0,
        quantile_alpha=0.95,
    )
    model = train_lightgbm(args, builder, train_arrays, val_arrays)
    predictions, targets, masks = predict_split(config, builder, model, "test_indices.csv", batch_rows=16)
    metrics = compute_patch_metrics(predictions, targets, masks, builder.target_settings, {"mae": AVAILABLE_METRICS["mae"]})

    assert predictions.shape == (1, 4, 4)
    assert np.isfinite(predictions).all()
    assert metrics["mae"] >= 0.0
