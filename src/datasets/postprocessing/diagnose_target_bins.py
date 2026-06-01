"""Report stitched prediction errors by target quantile and top-k bins."""

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
from src.datasets.dataset import get_test_dataloader
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
from src.datasets.utils import get_dataset_dimensions
from src.train import load_config
from src.trainer import Trainer
from src.utils import AVAILABLE_METRICS, seed_everything

DEFAULT_QUANTILE_EDGES = (0.0, 0.5, 0.75, 0.9, 0.95, 0.98, 0.99, 0.995, 1.0)
DEFAULT_TOP_FRACTIONS = (0.10, 0.05, 0.02, 0.01, 0.005)


def _resolve_split_name(config: Config, split: str) -> str:
    if split == "train":
        return config.data.train_split
    if split == "val":
        return config.data.val_split
    if split == "test":
        return config.data.test_split
    return split


def _normalize_hex_id(hex_id: object) -> str:
    if isinstance(hex_id, int | np.integer):
        return f"{int(hex_id):02d}"
    if isinstance(hex_id, float | np.floating) and np.isfinite(hex_id) and float(hex_id).is_integer():
        return f"{int(hex_id):02d}"
    return str(hex_id).zfill(2)


def _metric_summary(pred_values: np.ndarray, target_values: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    pred_values = pred_values[valid].astype(np.float64, copy=False)
    target_values = target_values[valid].astype(np.float64, copy=False)
    if target_values.size == 0:
        return {
            "count": 0.0,
            "target_mean": float("nan"),
            "pred_mean": float("nan"),
            "bias": float("nan"),
            "relative_bias": float("nan"),
            "mae": float("nan"),
            "rmse": float("nan"),
            "underprediction_fraction": float("nan"),
            "pred_target_ratio": float("nan"),
        }

    diff = pred_values - target_values
    target_mean = float(np.mean(target_values))
    pred_mean = float(np.mean(pred_values))
    bias = float(np.mean(diff))
    return {
        "count": float(target_values.size),
        "target_mean": target_mean,
        "pred_mean": pred_mean,
        "bias": bias,
        "relative_bias": bias / target_mean if abs(target_mean) > 1e-12 else float("nan"),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "underprediction_fraction": float(np.mean(diff < 0.0)),
        "pred_target_ratio": pred_mean / target_mean if abs(target_mean) > 1e-12 else float("nan"),
    }


def summarize_quantile_bins(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    quantile_edges: tuple[float, ...] = DEFAULT_QUANTILE_EDGES,
    thresholds: np.ndarray | None = None,
) -> pd.DataFrame:
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    pred_values = pred_values[valid]
    target_values = target_values[valid]
    if target_values.size == 0:
        return pd.DataFrame()

    edges = np.asarray(quantile_edges, dtype=np.float64)
    if thresholds is None:
        thresholds = np.nanquantile(target_values, edges)
    thresholds = np.asarray(thresholds, dtype=np.float64)

    rows = []
    for idx, (lo_q, hi_q) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        lo = thresholds[idx]
        hi = thresholds[idx + 1]
        selected = target_values >= lo if idx == 0 else target_values > lo
        selected &= target_values <= hi
        row = {
            "bin_type": "quantile",
            "bin": f"q{lo_q:.3f}-{hi_q:.3f}",
            "lower_quantile": float(lo_q),
            "upper_quantile": float(hi_q),
            "target_min": float(lo),
            "target_max": float(hi),
        }
        row.update(_metric_summary(pred_values[selected], target_values[selected]))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_topk_bins(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    top_fractions: tuple[float, ...] = DEFAULT_TOP_FRACTIONS,
    thresholds: dict[float, float] | None = None,
) -> pd.DataFrame:
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    pred_values = pred_values[valid]
    target_values = target_values[valid]
    if target_values.size == 0:
        return pd.DataFrame()

    thresholds = thresholds or {fraction: float(np.nanquantile(target_values, 1.0 - fraction)) for fraction in top_fractions}
    rows = []
    for fraction in top_fractions:
        threshold = thresholds[fraction]
        selected = target_values >= threshold
        row = {
            "bin_type": "topk",
            "bin": f"top{fraction:.3f}",
            "top_fraction": float(fraction),
            "target_threshold": float(threshold),
        }
        row.update(_metric_summary(pred_values[selected], target_values[selected]))
        rows.append(row)
    return pd.DataFrame(rows)


def _predict_split(config: Config, split: str, device: torch.device) -> tuple[np.ndarray, pd.DataFrame]:
    config.logger.enabled = False
    seed = getattr(config, "seed", 42)
    deterministic = getattr(config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    split_name = _resolve_split_name(config, split)
    original_test_split = config.data.test_split
    config.data.test_split = split_name
    try:
        loader = get_test_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=seed)
        spatial_channels, auxiliary_input_dims = get_dataset_dimensions(loader.dataset)
        trainer = Trainer(config, spatial_input_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims)
        trainer.device = device
        trainer.model.to(device)
        print(f"[target-bin diagnostics] Loading checkpoint {config.evaluation.checkpoint_filename}")
        checkpoint = trainer.load_model(filename=config.evaluation.checkpoint_filename, map_location=str(device))
        print(f"[target-bin diagnostics] Loaded epoch={checkpoint.get('epoch', 'N/A')} metrics={checkpoint.get('metric_value', 'N/A')}")
        _, predictions = trainer.test(loader, return_predictions=True)
        split_df = pd.read_csv(os.path.join(config.data.root_dir, split_name))
        split_df = split_df[split_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)
        return predictions, split_df
    finally:
        config.data.test_split = original_test_split


def _stitched_prediction_rows(
    config: Config,
    predictions: np.ndarray,
    split_df: pd.DataFrame,
    split: str,
    stitch_mode: str,
    center_crop_fraction: float,
    device: torch.device,
    mask_scope: str = "actual",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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

    split_df = split_df.copy()
    validate_patch_metadata_mask_scope(split_df, mask_scope)
    split_df["hex_id_norm"] = split_df["hex_id"].map(_normalize_hex_id)
    all_pred_values = []
    all_target_values = []
    hex_payloads = []
    hex_ids = sorted(split_df["hex_id_norm"].unique())

    for hex_id in hex_ids:
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
        valid = np.isfinite(pred_grid) & np.isfinite(target_grid) & (target_grid >= 0.0)
        pred_values = pred_grid[valid]
        target_values = target_grid[valid]
        all_pred_values.append(pred_values)
        all_target_values.append(target_values)
        hex_payloads.append((hex_id, pred_values, target_values))

        metrics = calculate_hexel_metrics_pytorch(
            gt_grid=target_grid,
            pred_grid=pred_grid,
            device=device,
            metric_functions=metric_functions,
        )
        print(
            f"[target-bin diagnostics] {target.name} hex{hex_id}: "
            f"mae={metrics.get('mae', float('nan')):.6g} "
            f"bias={metrics.get('bias', float('nan')):.6g} "
            f"ccc={metrics.get('ccc', float('nan')):.6g} "
            f"spearman={metrics.get('spearman', float('nan')):.6g}"
        )

    pred_all = np.concatenate(all_pred_values)
    target_all = np.concatenate(all_target_values)
    quantile_edges = np.asarray(DEFAULT_QUANTILE_EDGES, dtype=np.float64)
    thresholds = np.nanquantile(target_all, quantile_edges)
    top_thresholds = {fraction: float(np.nanquantile(target_all, 1.0 - fraction)) for fraction in DEFAULT_TOP_FRACTIONS}

    all_quantile = summarize_quantile_bins(pred_all, target_all, thresholds=thresholds)
    all_quantile.insert(0, "hex_id", "all")
    all_topk = summarize_topk_bins(pred_all, target_all, thresholds=top_thresholds)
    all_topk.insert(0, "hex_id", "all")

    per_hex_quantile = []
    per_hex_topk = []
    for hex_id, pred_values, target_values in hex_payloads:
        hex_quantile = summarize_quantile_bins(pred_values, target_values, thresholds=thresholds)
        if not hex_quantile.empty:
            hex_quantile.insert(0, "hex_id", f"hex{hex_id}")
            per_hex_quantile.append(hex_quantile)

        hex_topk = summarize_topk_bins(pred_values, target_values, thresholds=top_thresholds)
        if not hex_topk.empty:
            hex_topk.insert(0, "hex_id", f"hex{hex_id}")
            per_hex_topk.append(hex_topk)

    quantile_summary = pd.concat([all_quantile, *per_hex_quantile], ignore_index=True)
    topk_summary = pd.concat([all_topk, *per_hex_topk], ignore_index=True)
    for df in (quantile_summary, topk_summary):
        df.insert(0, "target", target.name)
        df.insert(1, "split", split)
        df.insert(2, "stitch_mode", stitch_mode)
        df.insert(3, "run", Path(config.save_dir).name)
    return (
        quantile_summary,
        topk_summary,
        pd.DataFrame({"target": target.name, "prediction_count": [float(pred_all.size)], "mask_scope": [mask_scope]}),
    )


def run_target_bin_diagnostics(
    config: Config,
    split: str,
    output_dir: Path,
    stitch_mode: str,
    center_crop_fraction: float,
    device: torch.device,
    mask_scope: str = "actual",
) -> dict[str, pd.DataFrame]:
    predictions, split_df = _predict_split(config=config, split=split, device=device)
    quantile_summary, topk_summary, metadata = _stitched_prediction_rows(
        config=config,
        predictions=predictions,
        split_df=split_df,
        split=split,
        stitch_mode=stitch_mode,
        center_crop_fraction=center_crop_fraction,
        device=device,
        mask_scope=mask_scope,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    quantile_path = output_dir / "bias_by_target_quantile.csv"
    topk_path = output_dir / "bias_by_target_topk.csv"
    metadata_path = output_dir / "metadata.csv"
    quantile_summary.to_csv(quantile_path, index=False)
    topk_summary.to_csv(topk_path, index=False)
    metadata.to_csv(metadata_path, index=False)
    print(f"[target-bin diagnostics] Wrote {quantile_path}")
    print(f"[target-bin diagnostics] Wrote {topk_path}")
    print("\n[All-hex quantile bins]")
    print(quantile_summary[quantile_summary["hex_id"] == "all"].to_string(index=False))
    print("\n[All-hex top-k bins]")
    print(topk_summary[topk_summary["hex_id"] == "all"].to_string(index=False))
    return {"quantile": quantile_summary, "topk": topk_summary, "metadata": metadata}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path, help="Config/checkpoint to evaluate.")
    parser.add_argument("--split", default="test", help="Split alias train/val/test or split CSV filename.")
    parser.add_argument("--output-dir", default=None, type=Path, help="Directory for CSV outputs.")
    parser.add_argument(
        "--stitch-mode",
        default="mean",
        choices=["mean", "max", "center_crop", "feathered", "non_overlap"],
        help="Stitching mode used before binning.",
    )
    parser.add_argument("--center-crop-fraction", type=float, default=0.8)
    parser.add_argument("--device", default=None, help="Torch device. Defaults to cuda if available else cpu.")
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    scope_dir = Path(config.save_dir) if args.mask_scope == "actual" else Path(config.save_dir) / f"{args.mask_scope}_mask_eval"
    output_dir = args.output_dir or scope_dir / "target_bin_diagnostics" / args.split / args.stitch_mode
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_target_bin_diagnostics(
        config=config,
        split=args.split,
        output_dir=output_dir,
        stitch_mode=args.stitch_mode,
        center_crop_fraction=args.center_crop_fraction,
        device=device,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()
