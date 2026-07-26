from unittest.mock import MagicMock

import pandas as pd
import pytest

import src.aggregate_multirun_results as aggregate_multirun_results
from src.aggregate_multirun_results import aggregate_eval_results, run_evaluations
from src.config import (
    Config,
    DataConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)


def _make_config(tmp_path) -> Config:
    return Config(
        save_dir=str(tmp_path / "out"),
        seed=42,
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"], checkpoint_filename="best.pth"),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
    )


def _write_test_metrics_csv(save_dir, run_id, seed, **metrics):
    save_dir.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "seed": seed, "save_dir": str(save_dir)}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(save_dir / "test_metrics.csv", index=False)
    return save_dir


def test_aggregate_eval_results_concatenates_and_appends_mean_std(tmp_path):
    save_dirs = [
        _write_test_metrics_csv(tmp_path / "seed_2", run_id=1, seed=2, **{"test_patch_mae": 0.2, "test_hexel/all/mae": 0.4}),
        _write_test_metrics_csv(tmp_path / "seed_1", run_id=0, seed=1, **{"test_patch_mae": 0.4, "test_hexel/all/mae": 0.6}),
    ]

    df = aggregate_eval_results([str(d) for d in save_dirs])

    # Sorted by seed ascending, with trailing mean/std summary rows.
    assert list(df["seed"][:2]) == [1, 2]
    assert list(df["seed"][2:]) == ["mean", "std"]

    mean_row = df[df["seed"] == "mean"].iloc[0]
    assert mean_row["test_patch_mae"] == pytest.approx(0.3)
    assert mean_row["test_hexel/all/mae"] == pytest.approx(0.5)
    # Non-numeric id columns are left blank/untouched in summary rows.
    assert mean_row["save_dir"] == ""


def test_aggregate_eval_results_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing test_metrics.csv"):
        aggregate_eval_results([str(tmp_path / "does_not_exist")])


def test_aggregate_eval_results_filters_by_metric_prefix(tmp_path):
    save_dirs = [
        _write_test_metrics_csv(
            tmp_path / "seed_1",
            run_id=0,
            seed=1,
            **{"test_patch_mae": 0.1, "test_patch_mse": 0.05, "test_hexel/all/mae": 0.2, "test_hexel/all/mse": 0.03},
        ),
    ]

    df = aggregate_eval_results([str(d) for d in save_dirs], metrics=["test_hexel/all/"])

    assert set(df.columns) == {"run_id", "seed", "save_dir", "test_hexel/all/mae", "test_hexel/all/mse"}


def test_aggregate_eval_results_unmatched_metric_raises(tmp_path):
    save_dirs = [_write_test_metrics_csv(tmp_path / "seed_1", run_id=0, seed=1, **{"test_patch_mae": 0.1})]

    with pytest.raises(ValueError, match="No matching columns found"):
        aggregate_eval_results([str(d) for d in save_dirs], metrics=["nonexistent_metric"])


def test_run_evaluations_skips_evaluate_hexels_when_test_metrics_csv_exists(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    monkeypatch.setattr(aggregate_multirun_results, "load_config", lambda path: config)

    seed_dir = tmp_path / "out" / "seed_42"
    _write_test_metrics_csv(seed_dir, run_id=0, seed=42, **{"test_patch_mae": 0.1})

    mock_run = MagicMock()
    monkeypatch.setattr(aggregate_multirun_results.subprocess, "run", mock_run)

    save_dirs = run_evaluations("unused.yaml", run_ids=[0], eval_args="")

    assert save_dirs == [str(seed_dir)]
    mock_run.assert_not_called()


def test_run_evaluations_runs_evaluate_hexels_when_test_metrics_csv_missing(tmp_path, monkeypatch):
    config = _make_config(tmp_path)
    monkeypatch.setattr(aggregate_multirun_results, "load_config", lambda path: config)

    mock_run = MagicMock()
    monkeypatch.setattr(aggregate_multirun_results.subprocess, "run", mock_run)

    save_dirs = run_evaluations("unused.yaml", run_ids=[0], eval_args="")

    assert save_dirs == [str(tmp_path / "out" / "seed_42")]
    mock_run.assert_called_once()
