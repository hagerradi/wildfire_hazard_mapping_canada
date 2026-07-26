"""
Aggregate multi-run (multi-seed) evaluation results into a single CSV.

Calls `python -m src.evaluate_hexels --config <config> --run_id <run_id>` once per run_id
(so evaluate_hexels.py itself stays a normal, standalone, run_id-driven CLI script), then
reads back the `eval_metrics.csv` each run writes to its own derived `save_dir` and
concatenates them into one aggregated CSV with trailing mean/std summary rows.

Usage:
    python -m src.aggregate_eval_results --config configs/bp_common_input_pipeline.yaml \
        --output_csv experiments/bp_common_input_pipeline_fuel_iROS/multi_run_eval_summary.csv
"""

import argparse
import os
import subprocess
import sys

import pandas as pd

from src.config import SEEDS, apply_run_id_overrides
from src.evaluate_hexels import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate multi-run (multi-seed) evaluation results into a CSV.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/bp_common_input_pipeline.yaml",
        help="Path to the base YAML config file (the same one used for training with run_files/train_no_tmp_copy_array.sh).",
    )
    parser.add_argument(
        "--run_ids",
        type=int,
        nargs="+",
        default=list(range(len(SEEDS))),
        help=f"run_ids to evaluate and aggregate (each must be in [0, {len(SEEDS) - 1}]). Defaults to all seeds.",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Path to write the aggregated CSV. Defaults to '<base config save_dir>/multi_run_eval_summary.csv'.",
    )
    parser.add_argument(
        "--eval_args",
        type=str,
        default="",
        help="Extra args forwarded as-is to each `src.evaluate_hexels` call, e.g. '--metrics_only --skip_hexel_plots'.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["test_hexel/all/", "patch/"],
        help="Metric columns to keep and summarize with mean/std, e.g. 'patch/mse test_hexel/all/mse'. Matches exact "
        "column names or prefixes (e.g. 'test_hexel/all/' keeps every column starting with it). Defaults to every "
        "numeric column (all patch/*, test_hexel/*, and val_hexel/* metrics) when omitted.",
    )
    return parser.parse_args()


def run_evaluations(config_path: str, run_ids: list[int], eval_args: str) -> list[str]:
    """
    Run `python -m src.evaluate_hexels --run_id=<id>` once per run_id, returning the list of
    save_dirs each run wrote its `eval_metrics.csv` to.
    """
    save_dirs = []
    for run_id in run_ids:
        config = load_config(config_path)
        apply_run_id_overrides(config, run_id)
        save_dirs.append(config.save_dir)

        print(f"\n=== Running evaluation for run_id={run_id} ===")
        cmd = [sys.executable, "-m", "src.evaluate_hexels", f"--config={config_path}", f"--run_id={run_id}", *eval_args.split()]
        subprocess.run(cmd, check=True)

    return save_dirs


def aggregate_eval_results(save_dirs: list[str], metrics: list[str] | None = None) -> pd.DataFrame:
    """
    Read each run's `eval_metrics.csv` (written by `src.evaluate_hexels`) from `save_dirs`,
    concatenate them into a single DataFrame, and append trailing "mean"/"std" summary rows.

    If `metrics` is given, only id columns (run_id, seed, save_dir) plus columns matching
    `metrics` (by exact name or prefix, e.g. "test_hexel/all/") are kept and summarized. Otherwise
    every numeric column is kept and summarized.
    """
    eval_csvs = [os.path.join(save_dir, "eval_metrics.csv") for save_dir in save_dirs]
    missing = [path for path in eval_csvs if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError(f"Missing eval_metrics.csv for run(s): {missing}")

    df = pd.concat((pd.read_csv(path) for path in eval_csvs), ignore_index=True).sort_values("seed").reset_index(drop=True)

    id_cols = ["run_id", "seed", "save_dir"]
    if metrics:
        keep_cols = id_cols + [col for col in df.columns if col not in id_cols and any(col == m or col.startswith(m) for m in metrics)]
        missing_metrics = [m for m in metrics if not any(col == m or col.startswith(m) for col in df.columns)]
        if missing_metrics:
            raise ValueError(f"No matching columns found for requested metrics: {missing_metrics}. Available columns: {list(df.columns)}")
        df = df[keep_cols]

    numeric_cols = df.select_dtypes(include="number").columns.difference(["run_id", "seed"])
    summary = df[numeric_cols].agg(["mean", "std"])
    summary.insert(0, "save_dir", "")
    summary.insert(0, "seed", ["mean", "std"])
    summary.insert(0, "run_id", ["mean", "std"])

    return pd.concat([df, summary], ignore_index=True)


def main() -> None:
    args = parse_args()

    base_config = load_config(args.config)
    output_csv = args.output_csv or os.path.join(base_config.save_dir, "multi_run_eval_summary.csv")

    save_dirs = run_evaluations(args.config, args.run_ids, args.eval_args)
    df = aggregate_eval_results(save_dirs, metrics=args.metrics)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nWrote aggregated eval results for {len(df) - 2} run(s) to: {output_csv}")


if __name__ == "__main__":
    main()
