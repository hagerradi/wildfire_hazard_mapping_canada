import pandas as pd
import pytest

from src.aggregate_multirun_results import aggregate_eval_results


def _write_eval_csv(save_dir, run_id, seed, **metrics):
    save_dir.mkdir(parents=True, exist_ok=True)
    row = {"run_id": run_id, "seed": seed, "save_dir": str(save_dir)}
    row.update(metrics)
    pd.DataFrame([row]).to_csv(save_dir / "eval_metrics.csv", index=False)
    return save_dir


def test_aggregate_eval_results_concatenates_and_appends_mean_std(tmp_path):
    save_dirs = [
        _write_eval_csv(tmp_path / "seed_2", run_id=1, seed=2, **{"patch/mae": 0.2, "test_hexel/all/mae": 0.4}),
        _write_eval_csv(tmp_path / "seed_1", run_id=0, seed=1, **{"patch/mae": 0.4, "test_hexel/all/mae": 0.6}),
    ]

    df = aggregate_eval_results([str(d) for d in save_dirs])

    # Sorted by seed ascending, with trailing mean/std summary rows.
    assert list(df["seed"][:2]) == [1, 2]
    assert list(df["seed"][2:]) == ["mean", "std"]

    mean_row = df[df["seed"] == "mean"].iloc[0]
    assert mean_row["patch/mae"] == pytest.approx(0.3)
    assert mean_row["test_hexel/all/mae"] == pytest.approx(0.5)
    # Non-numeric id columns are left blank/untouched in summary rows.
    assert mean_row["save_dir"] == ""


def test_aggregate_eval_results_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing eval_metrics.csv"):
        aggregate_eval_results([str(tmp_path / "does_not_exist")])


def test_aggregate_eval_results_filters_by_metric_prefix(tmp_path):
    save_dirs = [
        _write_eval_csv(
            tmp_path / "seed_1",
            run_id=0,
            seed=1,
            **{"patch/mae": 0.1, "patch/mse": 0.05, "test_hexel/all/mae": 0.2, "test_hexel/all/mse": 0.03},
        ),
    ]

    df = aggregate_eval_results([str(d) for d in save_dirs], metrics=["test_hexel/all/"])

    assert set(df.columns) == {"run_id", "seed", "save_dir", "test_hexel/all/mae", "test_hexel/all/mse"}


def test_aggregate_eval_results_unmatched_metric_raises(tmp_path):
    save_dirs = [_write_eval_csv(tmp_path / "seed_1", run_id=0, seed=1, **{"patch/mae": 0.1})]

    with pytest.raises(ValueError, match="No matching columns found"):
        aggregate_eval_results([str(d) for d in save_dirs], metrics=["nonexistent_metric"])
