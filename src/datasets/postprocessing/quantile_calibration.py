"""Fit validation quantile calibration and evaluate calibrated stitched hexels."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.spatial.utils import get_range_output
from src.config import Config
from src.datasets.postprocessing.diagnose_target_bins import (
    DEFAULT_QUANTILE_EDGES,
    DEFAULT_TOP_FRACTIONS,
    _normalize_hex_id,
    _predict_split,
    summarize_quantile_bins,
    summarize_topk_bins,
)
from src.datasets.postprocessing.utils import (
    as_float_array_with_nan,
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

DEFAULT_CALIBRATION_QUANTILES = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.02,
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.98,
    0.99,
    0.995,
    0.999,
    1.0,
)


@dataclass(frozen=True)
class QuantileMapping:
    table: pd.DataFrame
    knots: pd.DataFrame
    output_min: float | None = None
    output_max: float | None = None

    @property
    def pred_knots(self) -> np.ndarray:
        return self.knots["pred_value"].to_numpy(dtype=np.float64)

    @property
    def target_knots(self) -> np.ndarray:
        return self.knots["target_value"].to_numpy(dtype=np.float64)


@dataclass(frozen=True)
class HexPrediction:
    hex_id: str
    pred_grid: np.ndarray
    target_grid: np.ndarray
    valid_mask: np.ndarray

    @property
    def pred_values(self) -> np.ndarray:
        return self.pred_grid[self.valid_mask]

    @property
    def target_values(self) -> np.ndarray:
        return self.target_grid[self.valid_mask]


@dataclass(frozen=True)
class StitchedSplit:
    split: str
    target: str
    hex_predictions: list[HexPrediction]

    @property
    def pred_values(self) -> np.ndarray:
        return np.concatenate([item.pred_values for item in self.hex_predictions])

    @property
    def target_values(self) -> np.ndarray:
        return np.concatenate([item.target_values for item in self.hex_predictions])


def fit_quantile_mapping(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    quantiles: tuple[float, ...] = DEFAULT_CALIBRATION_QUANTILES,
    output_min: float | None = None,
    output_max: float | None = None,
) -> QuantileMapping:
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    pred_values = pred_values[valid].astype(np.float64, copy=False)
    target_values = target_values[valid].astype(np.float64, copy=False)
    if pred_values.size < 2:
        raise ValueError("At least two finite prediction/target pairs are required to fit quantile mapping.")

    q = np.asarray(quantiles, dtype=np.float64)
    if np.any(q < 0.0) or np.any(q > 1.0) or np.any(np.diff(q) <= 0.0):
        raise ValueError("Calibration quantiles must be strictly increasing in [0, 1].")

    pred_q = np.nanquantile(pred_values, q)
    target_q = np.nanquantile(target_values, q)
    table = pd.DataFrame({"quantile": q, "pred_value": pred_q, "target_value": target_q})

    unique_pred, inverse = np.unique(pred_q, return_inverse=True)
    if unique_pred.size < 2:
        raise ValueError("Prediction quantiles collapsed to one value; cannot fit a monotonic quantile map.")

    knot_targets = np.empty(unique_pred.shape, dtype=np.float64)
    for idx in range(unique_pred.size):
        knot_targets[idx] = float(np.mean(target_q[inverse == idx]))
    knot_targets = np.maximum.accumulate(knot_targets)
    if output_min is not None or output_max is not None:
        knot_targets = np.clip(
            knot_targets,
            -np.inf if output_min is None else output_min,
            np.inf if output_max is None else output_max,
        )

    knots = pd.DataFrame({"pred_value": unique_pred, "target_value": knot_targets})
    return QuantileMapping(table=table, knots=knots, output_min=output_min, output_max=output_max)


def apply_quantile_mapping(values: np.ndarray, mapping: QuantileMapping) -> np.ndarray:
    calibrated = values.astype(np.float32, copy=True)
    valid = np.isfinite(calibrated)
    if not valid.any():
        return calibrated

    calibrated_values = np.interp(
        calibrated[valid].astype(np.float64, copy=False),
        mapping.pred_knots,
        mapping.target_knots,
        left=mapping.target_knots[0],
        right=mapping.target_knots[-1],
    )
    if mapping.output_min is not None or mapping.output_max is not None:
        calibrated_values = np.clip(
            calibrated_values,
            -np.inf if mapping.output_min is None else mapping.output_min,
            np.inf if mapping.output_max is None else mapping.output_max,
        )
    calibrated[valid] = calibrated_values.astype(np.float32)
    return calibrated


def _collect_stitched_split(
    config: Config,
    split: str,
    stitch_mode: str,
    center_crop_fraction: float,
    device: torch.device,
    mask_scope: str = "actual",
) -> StitchedSplit:
    predictions, split_df = _predict_split(config=config, split=split, device=device)
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

    split_df = split_df.copy()
    validate_patch_metadata_mask_scope(split_df, mask_scope)
    split_df["hex_id_norm"] = split_df["hex_id"].map(_normalize_hex_id)
    hex_predictions = []

    for hex_id in sorted(split_df["hex_id_norm"].unique()):
        hex_df = split_df[split_df["hex_id_norm"] == hex_id]
        hex_indices = hex_df.index.tolist()
        reconstructed, profile = get_predicted_hexel(
            base_dir=config.data.root_dir,
            raw_data_dir=config.data.raw_data_dir,
            test_df=hex_df,
            predictions=predictions[hex_indices],
            min_target_val=min_target_val,
            max_target_val=max_target_val,
            hex_id=hex_id,
            modelling_approach=config.modelling_approach,
            out_norm=out_norm,
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
            stitch_mode=stitch_mode,
            target_channel_index=target_channel_index,
            prediction_mask_channel_indices=prediction_mask_channel_indices,
            win_h=config.data_prep.win_h,
            win_w=config.data_prep.win_w,
            center_crop_fraction=center_crop_fraction,
            mask_scope=mask_scope,
        )
        paths = Paths(hex_id=hex_id, root_dir=config.data.raw_data_dir)
        target_grid, reconstructed = load_target_grid_for_mask_scope(
            paths=paths,
            target=target,
            pred_grid=reconstructed,
            profile=profile,
            mask_scope=mask_scope,
            hex_id=hex_id,
        )
        target_grid = as_float_array_with_nan(target_grid)
        pred_grid = as_float_array_with_nan(reconstructed)
        valid_mask = np.isfinite(pred_grid) & np.isfinite(target_grid) & (target_grid >= 0.0)
        hex_predictions.append(
            HexPrediction(
                hex_id=f"hex{hex_id}",
                pred_grid=pred_grid,
                target_grid=target_grid,
                valid_mask=valid_mask,
            )
        )

    return StitchedSplit(split=split, target=target.name, hex_predictions=hex_predictions)


def _metric_rows(
    split_data: StitchedSplit,
    metric_functions: dict[str, object],
    device: torch.device,
    mapping: QuantileMapping | None,
) -> pd.DataFrame:
    rows = []
    metrics_by_stage: dict[str, dict[str, list[float]]] = {}
    for stage in ("uncalibrated", "calibrated"):
        if stage == "calibrated" and mapping is None:
            continue
        metrics_by_stage[stage] = {}
        for item in split_data.hex_predictions:
            pred_grid = item.pred_grid if stage == "uncalibrated" else apply_quantile_mapping(item.pred_grid, mapping)
            metrics = calculate_hexel_metrics_pytorch(
                gt_grid=item.target_grid,
                pred_grid=pred_grid,
                device=device,
                metric_functions=metric_functions,
            )
            row = {"split": split_data.split, "target": split_data.target, "stage": stage, "hex_id": item.hex_id}
            row.update(metrics)
            rows.append(row)
            for key, value in metrics.items():
                metrics_by_stage[stage].setdefault(key, []).append(value)

    for stage, stage_metrics in metrics_by_stage.items():
        row = {"split": split_data.split, "target": split_data.target, "stage": stage, "hex_id": "all"}
        for key, values in stage_metrics.items():
            finite_values = [float(value) for value in values if np.isfinite(value)]
            row[key] = float(np.mean(finite_values)) if finite_values else float("nan")
        rows.append(row)

    return pd.DataFrame(rows)


def _bin_summary_rows(split_data: StitchedSplit, mapping: QuantileMapping | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    target_all = split_data.target_values
    quantile_thresholds = np.nanquantile(target_all, np.asarray(DEFAULT_QUANTILE_EDGES, dtype=np.float64))
    top_thresholds = {fraction: float(np.nanquantile(target_all, 1.0 - fraction)) for fraction in DEFAULT_TOP_FRACTIONS}

    quantile_frames = []
    topk_frames = []
    for stage in ("uncalibrated", "calibrated"):
        if stage == "calibrated" and mapping is None:
            continue
        for hex_id, pred_values, target_values in _iter_stage_values(split_data, stage=stage, mapping=mapping):
            quantile = summarize_quantile_bins(pred_values, target_values, thresholds=quantile_thresholds)
            topk = summarize_topk_bins(pred_values, target_values, thresholds=top_thresholds)
            if not quantile.empty:
                quantile.insert(0, "hex_id", hex_id)
                quantile_frames.append(quantile)
            if not topk.empty:
                topk.insert(0, "hex_id", hex_id)
                topk_frames.append(topk)

    quantile_summary = pd.concat(quantile_frames, ignore_index=True)
    topk_summary = pd.concat(topk_frames, ignore_index=True)
    for df in (quantile_summary, topk_summary):
        df.insert(0, "split", split_data.split)
        df.insert(1, "target", split_data.target)
    return quantile_summary, topk_summary


def _iter_stage_values(
    split_data: StitchedSplit,
    stage: str,
    mapping: QuantileMapping | None,
):
    all_pred = []
    all_target = []
    for item in split_data.hex_predictions:
        pred_grid = item.pred_grid if stage == "uncalibrated" else apply_quantile_mapping(item.pred_grid, mapping)
        pred_values = pred_grid[item.valid_mask]
        target_values = item.target_values
        all_pred.append(pred_values)
        all_target.append(target_values)
        yield item.hex_id, pred_values, target_values

    yield "all", np.concatenate(all_pred), np.concatenate(all_target)


def run_quantile_calibration(
    config: Config,
    output_dir: Path,
    calibration_split: str,
    apply_splits: list[str],
    stitch_mode: str,
    center_crop_fraction: float,
    device: torch.device,
    mask_scope: str = "actual",
) -> dict[str, pd.DataFrame]:
    target = get_config_target_spec(config)
    max_target_val, min_target_val = get_range_output(root_dir=config.data.raw_data_dir, output_type=target.output_type)
    output_min = max(0.0, float(min_target_val))
    output_max = float(max_target_val) if target.probability_scale else None
    if target.probability_scale:
        output_max = min(1.0, output_max)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"[quantile calibration] Collecting calibration split={calibration_split}")
    calibration_data = _collect_stitched_split(
        config=config,
        split=calibration_split,
        stitch_mode=stitch_mode,
        center_crop_fraction=center_crop_fraction,
        device=device,
        mask_scope=mask_scope,
    )
    mapping = fit_quantile_mapping(
        pred_values=calibration_data.pred_values,
        target_values=calibration_data.target_values,
        output_min=output_min,
        output_max=output_max,
    )
    mapping.table.to_csv(output_dir / "quantile_mapping_table.csv", index=False)
    mapping.knots.to_csv(output_dir / "quantile_mapping_knots.csv", index=False)

    metric_functions = {name: AVAILABLE_METRICS[name] for name in config.metrics}
    split_cache = {calibration_split: calibration_data}
    metrics_frames = []
    quantile_frames = []
    topk_frames = []
    for split in apply_splits:
        if split not in split_cache:
            print(f"[quantile calibration] Collecting apply split={split}")
            split_cache[split] = _collect_stitched_split(
                config=config,
                split=split,
                stitch_mode=stitch_mode,
                center_crop_fraction=center_crop_fraction,
                device=device,
                mask_scope=mask_scope,
            )
        split_data = split_cache[split]
        metrics_frames.append(_metric_rows(split_data, metric_functions=metric_functions, device=device, mapping=mapping))
        quantile_summary, topk_summary = _bin_summary_rows(split_data, mapping=mapping)
        quantile_frames.append(quantile_summary)
        topk_frames.append(topk_summary)

    metrics = pd.concat(metrics_frames, ignore_index=True)
    quantile_bins = pd.concat(quantile_frames, ignore_index=True)
    topk_bins = pd.concat(topk_frames, ignore_index=True)
    metadata = pd.DataFrame(
        [
            {
                "target": target.name,
                "run": Path(config.save_dir).name,
                "calibration_split": calibration_split,
                "apply_splits": ",".join(apply_splits),
                "stitch_mode": stitch_mode,
                "output_min": output_min,
                "output_max": output_max,
                "mask_scope": mask_scope,
            }
        ]
    )
    metrics.to_csv(output_dir / "calibrated_metrics.csv", index=False)
    quantile_bins.to_csv(output_dir / "calibrated_bias_by_target_quantile.csv", index=False)
    topk_bins.to_csv(output_dir / "calibrated_bias_by_target_topk.csv", index=False)
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"[quantile calibration] Wrote outputs to {output_dir}")
    for split in apply_splits:
        all_rows = metrics[(metrics["split"] == split) & (metrics["hex_id"] == "all")]
        print(f"\n[quantile calibration] {split} all metrics")
        print(all_rows.to_string(index=False))
    return {"metrics": metrics, "quantile_bins": quantile_bins, "topk_bins": topk_bins, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--apply-splits", nargs="+", default=["val", "test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--stitch-mode",
        default="mean",
        choices=["mean", "max", "center_crop", "feathered", "non_overlap"],
    )
    parser.add_argument("--center-crop-fraction", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    scope_dir = Path(config.save_dir) if args.mask_scope == "actual" else Path(config.save_dir) / f"{args.mask_scope}_mask_eval"
    output_dir = args.output_dir or scope_dir / "quantile_calibration" / f"fit_{args.calibration_split}" / args.stitch_mode
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_quantile_calibration(
        config=config,
        output_dir=output_dir,
        calibration_split=args.calibration_split,
        apply_splits=args.apply_splits,
        stitch_mode=args.stitch_mode,
        center_crop_fraction=args.center_crop_fraction,
        device=device,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()
