"""Compare stitch modes for saved patch predictions without retraining."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.spatial.utils import get_range_output
from src.config import Config
from src.datasets.postprocessing.utils import (
    calculate_hexel_metrics_pytorch,
    get_config_grid_params,
    get_config_target_spec,
    get_predicted_hexel,
    get_prediction_mask_channel_indices,
    get_target_channel_index,
    get_target_log_stats,
    get_target_out_norm,
    load_target_grid_for_mask_scope,
    validate_patch_metadata_mask_scope,
)
from src.train import load_config
from src.utils import AVAILABLE_METRICS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Config used to generate the predictions.")
    parser.add_argument("--predictions", required=True, type=Path, help="Saved test_predictions.npy file.")
    parser.add_argument("--output-csv", required=True, type=Path, help="Where to write the comparison CSV.")
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["mean", "max", "center_crop", "feathered"],
        help="Stitch modes to evaluate.",
    )
    parser.add_argument("--center-crop-fraction", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def _hex_id_str(hex_id: object) -> str:
    return str(hex_id).zfill(2)


def _load_filtered_test_df(config: Config) -> pd.DataFrame:
    test_df = pd.read_csv(os.path.join(config.data.root_dir, config.data.test_split))
    return test_df[test_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)


def _aggregate_metric_rows(
    rows: list[dict[str, object]], metric_names: list[str], run_name: str, mode: str, mask_scope: str
) -> dict[str, object]:
    aggregate: dict[str, object] = {"run": run_name, "mode": mode, "hex_id": "all", "mask_scope": mask_scope}
    for metric_name in metric_names:
        values = []
        for row in rows:
            value = row.get(metric_name)
            if isinstance(value, int | float | np.floating) and np.isfinite(float(value)):
                values.append(float(value))
        aggregate[metric_name] = float(np.mean(values)) if values else float("nan")
    return aggregate


def evaluate_modes(
    config: Config,
    predictions: np.ndarray,
    modes: list[str],
    center_crop_fraction: float,
    device: torch.device,
    run_name: str,
    mask_scope: str = "actual",
) -> pd.DataFrame:
    if predictions.ndim == 4 and predictions.shape[1] == 1:
        predictions = predictions[:, 0]
    if predictions.ndim != 3:
        raise ValueError(f"Expected single-target predictions with shape (N,H,W) or (N,1,H,W), got {predictions.shape}.")

    target = get_config_target_spec(config)
    grid_params = get_config_grid_params(config)
    out_norm = get_target_out_norm(grid_params=grid_params, target=target, fallback_out_norm="min_max")
    target_log_mean, target_log_std = get_target_log_stats(grid_params=grid_params, target=target)
    max_target_val, min_target_val = get_range_output(root_dir=config.data.raw_data_dir, output_type=target.output_type)
    prediction_mask_channel_indices = get_prediction_mask_channel_indices(
        data_dir=config.data.root_dir,
        modelling_approach=config.modelling_approach,
        grid_params=grid_params,
    )
    target_channel_index = get_target_channel_index(
        data_dir=config.data.root_dir,
        modelling_approach=config.modelling_approach,
        target=target,
    )
    metric_functions = {name: AVAILABLE_METRICS[name] for name in config.metrics}

    test_df = _load_filtered_test_df(config)
    validate_patch_metadata_mask_scope(test_df, mask_scope)
    rows: list[dict[str, object]] = []
    for mode in modes:
        mode_rows: list[dict[str, object]] = []
        for hex_id in test_df["hex_id"].drop_duplicates():
            hex_df = test_df[test_df["hex_id"] == hex_id]
            hex_indices = hex_df.index.tolist()
            hex_id_norm = _hex_id_str(hex_id)
            reconstructed, profile = get_predicted_hexel(
                base_dir=config.data.root_dir,
                raw_data_dir=config.data.raw_data_dir,
                test_df=hex_df,
                predictions=predictions[hex_indices],
                min_target_val=min_target_val,
                max_target_val=max_target_val,
                hex_id=hex_id_norm,
                modelling_approach=config.modelling_approach,
                out_norm=out_norm,
                target_log_mean=target_log_mean,
                target_log_std=target_log_std,
                stitch_mode=mode,
                target_channel_index=target_channel_index,
                prediction_mask_channel_indices=prediction_mask_channel_indices,
                win_h=config.data_prep.win_h,
                win_w=config.data_prep.win_w,
                center_crop_fraction=center_crop_fraction,
                mask_scope=mask_scope,
            )
            paths = Paths(hex_id=hex_id_norm, root_dir=config.data.raw_data_dir)
            target_grid, reconstructed = load_target_grid_for_mask_scope(
                paths=paths,
                target=target,
                pred_grid=reconstructed,
                profile=profile,
                mask_scope=mask_scope,
                hex_id=hex_id_norm,
            )
            metrics = calculate_hexel_metrics_pytorch(
                gt_grid=target_grid,
                pred_grid=reconstructed,
                device=device,
                metric_functions=metric_functions,
            )
            row: dict[str, object] = {"run": run_name, "mode": mode, "hex_id": f"hex{hex_id_norm}", "mask_scope": mask_scope}
            row.update(metrics)
            mode_rows.append(row)
            rows.append(row)
        rows.append(_aggregate_metric_rows(mode_rows, list(metric_functions), run_name=run_name, mode=mode, mask_scope=mask_scope))
    result = pd.DataFrame(rows)
    if not result.empty and "mask_scope" not in result.columns:
        result["mask_scope"] = mask_scope
    return result


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    predictions = np.load(args.predictions)
    device = torch.device(args.device)
    run_name = args.predictions.parent.name
    summary = evaluate_modes(
        config=config,
        predictions=predictions,
        modes=args.modes,
        center_crop_fraction=args.center_crop_fraction,
        device=device,
        run_name=run_name,
        mask_scope=args.mask_scope,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_csv, index=False)
    print(summary[summary["hex_id"] == "all"].to_string(index=False))
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
