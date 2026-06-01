"""Diagnose stitched prediction dynamic range, coarse ranking, and calibrated threshold behavior."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.patches import Patch

from data_preparation.paths import MASK_SCOPE_CHOICES
from src.config import Config
from src.datasets.postprocessing.quantile_calibration import (
    QuantileMapping,
    _collect_stitched_split,
    apply_quantile_mapping,
)
from src.datasets.postprocessing.utils import get_config_target_spec, get_hexel_binary_maps, get_mask_scope_save_dir
from src.datasets.postprocessing.visualize_predictions import (
    plot_hexbin_distribution,
    plot_histogram_distribution,
    visualize_hexel_iou,
    visualize_target_grids,
)
from src.train import load_config

DEFAULT_SPREAD_QUANTILES = (
    0.0,
    0.001,
    0.005,
    0.01,
    0.05,
    0.10,
    0.25,
    0.50,
    0.75,
    0.90,
    0.95,
    0.99,
    0.995,
    0.999,
    1.0,
)
DEFAULT_BIN_COUNTS = (5, 10, 20)
DEFAULT_THRESHOLD_QUANTILES = (0.90, 0.95, 0.98, 0.99, 0.995)
DEFAULT_IOU_PERCENTILES = (0.995, 0.99, 0.98, 0.95, 0.90)


def _safe_ratio(numerator: float, denominator: float, eps: float = 1e-12) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or abs(denominator) <= eps:
        return float("nan")
    return float(numerator / denominator)


def _finite_pairs(pred_values: np.ndarray, target_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(pred_values, dtype=np.float64).reshape(-1)
    target = np.asarray(target_values, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred) & np.isfinite(target)
    return pred[valid], target[valid]


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(x_rank, y_rank)


def summarize_dynamic_range(pred_values: np.ndarray, target_values: np.ndarray) -> dict[str, float]:
    """Summarize absolute and relative dynamic range of predictions vs targets."""
    pred, target = _finite_pairs(pred_values, target_values)
    if pred.size == 0:
        return {"count": 0.0}

    pred_q = np.nanquantile(pred, DEFAULT_SPREAD_QUANTILES)
    target_q = np.nanquantile(target, DEFAULT_SPREAD_QUANTILES)
    pred_by_q = dict(zip(DEFAULT_SPREAD_QUANTILES, pred_q, strict=True))
    target_by_q = dict(zip(DEFAULT_SPREAD_QUANTILES, target_q, strict=True))

    abs_error = np.abs(pred - target)
    target_iqr = float(target_by_q[0.75] - target_by_q[0.25])
    pred_iqr = float(pred_by_q[0.75] - pred_by_q[0.25])
    target_p90_p10 = float(target_by_q[0.90] - target_by_q[0.10])
    pred_p90_p10 = float(pred_by_q[0.90] - pred_by_q[0.10])
    target_p99_p01 = float(target_by_q[0.99] - target_by_q[0.01])
    pred_p99_p01 = float(pred_by_q[0.99] - pred_by_q[0.01])
    target_p99_p50 = float(target_by_q[0.99] - target_by_q[0.50])
    pred_p99_p50 = float(pred_by_q[0.99] - pred_by_q[0.50])
    target_tail_spread = float(target_by_q[0.995] - target_by_q[0.95])
    pred_tail_spread = float(pred_by_q[0.995] - pred_by_q[0.95])
    mae = float(np.mean(abs_error))

    return {
        "count": float(pred.size),
        "target_mean": float(np.mean(target)),
        "pred_mean": float(np.mean(pred)),
        "bias": float(np.mean(pred - target)),
        "mae": mae,
        "rmse": float(np.sqrt(np.mean((pred - target) ** 2))),
        "target_std": float(np.std(target)),
        "pred_std": float(np.std(pred)),
        "pred_target_std_ratio": _safe_ratio(float(np.std(pred)), float(np.std(target))),
        "target_iqr": target_iqr,
        "pred_iqr": pred_iqr,
        "pred_target_iqr_ratio": _safe_ratio(pred_iqr, target_iqr),
        "mae_target_iqr_ratio": _safe_ratio(mae, target_iqr),
        "mae_pred_iqr_ratio": _safe_ratio(mae, pred_iqr),
        "target_p90_p10": target_p90_p10,
        "pred_p90_p10": pred_p90_p10,
        "pred_target_p90_p10_ratio": _safe_ratio(pred_p90_p10, target_p90_p10),
        "mae_target_p90_p10_ratio": _safe_ratio(mae, target_p90_p10),
        "target_p99_p01": target_p99_p01,
        "pred_p99_p01": pred_p99_p01,
        "pred_target_p99_p01_ratio": _safe_ratio(pred_p99_p01, target_p99_p01),
        "target_p99_p50": target_p99_p50,
        "pred_p99_p50": pred_p99_p50,
        "pred_target_p99_p50_ratio": _safe_ratio(pred_p99_p50, target_p99_p50),
        "target_p995_p95": target_tail_spread,
        "pred_p995_p95": pred_tail_spread,
        "pred_target_p995_p95_ratio": _safe_ratio(pred_tail_spread, target_tail_spread),
        "pred_unique_fraction": _safe_ratio(float(np.unique(pred).size), float(pred.size)),
    }


def summarize_percentiles(pred_values: np.ndarray, target_values: np.ndarray) -> pd.DataFrame:
    pred, target = _finite_pairs(pred_values, target_values)
    if pred.size == 0:
        return pd.DataFrame()

    pred_q = np.nanquantile(pred, DEFAULT_SPREAD_QUANTILES)
    target_q = np.nanquantile(target, DEFAULT_SPREAD_QUANTILES)
    rows = []
    for quantile, pred_value, target_value in zip(DEFAULT_SPREAD_QUANTILES, pred_q, target_q, strict=True):
        rows.append(
            {
                "quantile": quantile,
                "target_value": float(target_value),
                "pred_value": float(pred_value),
                "pred_minus_target": float(pred_value - target_value),
                "pred_target_ratio": _safe_ratio(float(pred_value), float(target_value)),
            }
        )
    return pd.DataFrame(rows)


def summarize_bin_ranking(pred_values: np.ndarray, target_values: np.ndarray, bin_count: int) -> tuple[dict[str, float], pd.DataFrame]:
    """Compare mean predictions across coarse target quantile bins."""
    pred, target = _finite_pairs(pred_values, target_values)
    base_summary = {"requested_bin_count": float(bin_count), "count": float(pred.size)}
    if pred.size < 2:
        return base_summary | {"actual_bin_count": 0.0}, pd.DataFrame()

    bins = pd.qcut(pd.Series(target), q=bin_count, labels=False, duplicates="drop")
    frame = pd.DataFrame({"target_bin": bins, "target": target, "pred": pred}).dropna(subset=["target_bin"])
    if frame.empty:
        return base_summary | {"actual_bin_count": 0.0}, pd.DataFrame()

    frame["target_bin"] = frame["target_bin"].astype(int)
    grouped = (
        frame.groupby("target_bin", as_index=False)
        .agg(
            count=("target", "size"),
            target_min=("target", "min"),
            target_max=("target", "max"),
            target_mean=("target", "mean"),
            target_median=("target", "median"),
            pred_mean=("pred", "mean"),
            pred_median=("pred", "median"),
            mae=("pred", lambda values: float(np.mean(np.abs(values.to_numpy() - frame.loc[values.index, "target"].to_numpy())))),
        )
        .sort_values("target_bin")
        .reset_index(drop=True)
    )

    target_means = grouped["target_mean"].to_numpy(dtype=np.float64)
    pred_means = grouped["pred_mean"].to_numpy(dtype=np.float64)
    pair_count = 0
    pair_score = 0.0
    for left in range(pred_means.size):
        for right in range(left + 1, pred_means.size):
            pair_count += 1
            if pred_means[right] > pred_means[left]:
                pair_score += 1.0
            elif pred_means[right] == pred_means[left]:
                pair_score += 0.5

    pred_bin_spread = float(np.max(pred_means) - np.min(pred_means)) if pred_means.size else float("nan")
    target_bin_spread = float(np.max(target_means) - np.min(target_means)) if target_means.size else float("nan")
    summary = base_summary | {
        "actual_bin_count": float(len(grouped)),
        "bin_mean_spearman": _spearman_corr(target_means, pred_means),
        "bin_mean_pearson": _pearson_corr(target_means, pred_means),
        "bin_pair_order_accuracy": _safe_ratio(pair_score, float(pair_count)),
        "adjacent_bin_monotone_fraction": float(np.mean(np.diff(pred_means) >= 0.0)) if pred_means.size > 1 else float("nan"),
        "target_bin_mean_spread": target_bin_spread,
        "pred_bin_mean_spread": pred_bin_spread,
        "pred_target_bin_mean_spread_ratio": _safe_ratio(pred_bin_spread, target_bin_spread),
    }
    return summary, grouped


def summarize_threshold_exceedance(
    pred_values: np.ndarray,
    target_values: np.ndarray,
    thresholds: dict[float, float],
) -> pd.DataFrame:
    """Evaluate hotspot maps using fixed target-value thresholds instead of top-k rank thresholds."""
    pred, target = _finite_pairs(pred_values, target_values)
    rows = []
    for target_quantile, threshold in thresholds.items():
        if pred.size == 0:
            rows.append({"target_quantile": target_quantile, "threshold": threshold, "count": 0.0})
            continue
        pred_bin = pred >= threshold
        target_bin = target >= threshold
        intersection = int(np.sum(pred_bin & target_bin))
        union = int(np.sum(pred_bin | target_bin))
        pred_count = int(np.sum(pred_bin))
        target_count = int(np.sum(target_bin))
        rows.append(
            {
                "target_quantile": target_quantile,
                "threshold": threshold,
                "count": float(pred.size),
                "target_count": float(target_count),
                "pred_count": float(pred_count),
                "target_fraction": _safe_ratio(float(target_count), float(pred.size)),
                "pred_fraction": _safe_ratio(float(pred_count), float(pred.size)),
                "pred_target_area_ratio": _safe_ratio(float(pred_count), float(target_count)),
                "intersection": float(intersection),
                "union": float(union),
                "precision": _safe_ratio(float(intersection), float(pred_count)),
                "recall": _safe_ratio(float(intersection), float(target_count)),
                "iou": _safe_ratio(float(intersection), float(union)),
            }
        )
    return pd.DataFrame(rows)


def fixed_threshold_binary_maps(
    pred_grid: np.ndarray,
    target_grid: np.ndarray,
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred = np.asarray(pred_grid, dtype=np.float32)
    target = np.asarray(target_grid, dtype=np.float32)
    valid_mask = np.isfinite(pred) & np.isfinite(target)

    pred_bin = np.zeros(pred.shape, dtype=bool)
    target_bin = np.zeros(target.shape, dtype=bool)
    pred_bin[valid_mask] = pred[valid_mask] >= threshold
    target_bin[valid_mask] = target[valid_mask] >= threshold
    return pred_bin, target_bin, valid_mask


def _binary_overlap_stats(pred_bin: np.ndarray, target_bin: np.ndarray, valid_mask: np.ndarray) -> dict[str, float]:
    pred_valid = pred_bin & valid_mask
    target_valid = target_bin & valid_mask
    intersection = float(np.sum(pred_valid & target_valid))
    union = float(np.sum(pred_valid | target_valid))
    pred_count = float(np.sum(pred_valid))
    target_count = float(np.sum(target_valid))
    valid_count = float(np.sum(valid_mask))
    return {
        "pred_count": pred_count,
        "target_count": target_count,
        "pred_fraction": _safe_ratio(pred_count, valid_count),
        "target_fraction": _safe_ratio(target_count, valid_count),
        "area_ratio": _safe_ratio(pred_count, target_count),
        "precision": _safe_ratio(intersection, pred_count),
        "recall": _safe_ratio(intersection, target_count),
        "iou": _safe_ratio(intersection, union),
    }


def _threshold_label(target_quantile: float) -> str:
    return f"q{target_quantile:g}".replace(".", "p")


def normalize_threshold_quantiles(values: list[float] | tuple[float, ...]) -> tuple[float, ...]:
    quantiles = []
    for value in values:
        quantile = float(value)
        if quantile > 1.0:
            quantile /= 100.0
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"threshold quantiles must be in (0, 1) or percent form in (0, 100), got {value}.")
        quantiles.append(quantile)
    return tuple(quantiles)


def visualize_fixed_threshold_exceedance(
    target_grid: np.ndarray,
    pred_grid: np.ndarray,
    threshold: float,
    target_quantile: float,
    hex_id: str,
    save_dir: Path,
    stage: str,
    target_label: str,
    threshold_scope: str,
) -> None:
    pred_bin, target_bin, valid_mask = fixed_threshold_binary_maps(pred_grid, target_grid, threshold)
    stats = _binary_overlap_stats(pred_bin, target_bin, valid_mask)

    pred = np.asarray(pred_grid, dtype=np.float32)
    target = np.asarray(target_grid, dtype=np.float32)
    pred_display = np.where(valid_mask, pred, np.nan)
    target_display = np.where(valid_mask, target, np.nan)
    finite_values = np.concatenate([pred_display[valid_mask], target_display[valid_mask]])
    vmax = float(np.nanmax(finite_values)) if finite_values.size else threshold
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = max(float(threshold), 1.0)

    height, width = target.shape
    rgb_overlap = np.ones((height, width, 3), dtype=np.float32) * 0.95
    pred_valid = pred_bin & valid_mask
    target_valid = target_bin & valid_mask
    rgb_overlap[pred_valid & ~target_valid] = [1.0, 0.0, 0.0]
    rgb_overlap[~pred_valid & target_valid] = [0.0, 0.0, 1.0]
    rgb_overlap[pred_valid & target_valid] = [1.0, 0.0, 1.0]
    rgb_overlap[~valid_mask] = [1.0, 1.0, 1.0]

    quantile_label = _threshold_label(target_quantile)
    out_dir = save_dir / "predicted_hexels_plot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"hexel_{hex_id}_{stage}_fixed_threshold_{quantile_label}.png"

    fig, axes = plt.subplots(2, 2, figsize=(14, 12), layout="constrained")
    fig.suptitle(
        f"{threshold_scope.replace('_', '-').title()} Target-Threshold Exceedance - "
        f"Hex {hex_id} - {stage} - {quantile_label} threshold={threshold:.6g}",
        fontsize=16,
    )

    axes[0, 0].imshow(pred_display, cmap="viridis", origin="upper", vmin=0.0, vmax=vmax)
    axes[0, 0].contour(np.nan_to_num(pred_bin), levels=[0.5], colors="white", linewidths=0.4, alpha=0.8)
    axes[0, 0].set_title("Prediction with fixed-threshold contour")

    im = axes[0, 1].imshow(target_display, cmap="viridis", origin="upper", vmin=0.0, vmax=vmax)
    axes[0, 1].contour(np.nan_to_num(target_bin), levels=[0.5], colors="white", linewidths=0.4, alpha=0.8)
    axes[0, 1].set_title("Ground truth with same fixed threshold")
    fig.colorbar(im, ax=axes[0, 1], label=target_label, shrink=0.8)

    axes[1, 0].imshow(rgb_overlap, origin="upper")
    axes[1, 0].set_title("Fixed-threshold overlap")
    axes[1, 0].legend(
        handles=[
            Patch(facecolor="magenta", edgecolor="black", label="Intersection"),
            Patch(facecolor="red", edgecolor="black", label="Prediction only"),
            Patch(facecolor="blue", edgecolor="black", label="Ground truth only"),
        ],
        loc="upper right",
        framealpha=0.9,
        fontsize=10,
    )

    stats_text = "\n".join(
        [
            f"threshold scope: {threshold_scope}",
            f"target quantile: {target_quantile:g}",
            f"threshold: {threshold:.6g}",
            f"pred area: {stats['pred_fraction']:.4f}",
            f"target area: {stats['target_fraction']:.4f}",
            f"area ratio: {stats['area_ratio']:.4f}",
            f"precision: {stats['precision']:.4f}",
            f"recall: {stats['recall']:.4f}",
            f"IoU: {stats['iou']:.4f}",
        ]
    )
    axes[1, 1].axis("off")
    axes[1, 1].text(0.0, 1.0, stats_text, va="top", ha="left", fontsize=13, family="monospace")

    for ax in axes.flat:
        if ax.has_data():
            ax.set_xlabel("Easting (m)")
            ax.set_ylabel("Northing (m)")

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_quantile_mapping(mapping_dir: Path) -> QuantileMapping:
    knots_path = mapping_dir / "quantile_mapping_knots.csv"
    table_path = mapping_dir / "quantile_mapping_table.csv"
    metadata_path = mapping_dir / "metadata.csv"
    if not knots_path.exists():
        raise FileNotFoundError(f"Missing quantile mapping knots: {knots_path}")

    knots = pd.read_csv(knots_path)
    table = pd.read_csv(table_path) if table_path.exists() else knots.copy()
    output_min = None
    output_max = None
    if metadata_path.exists():
        metadata = pd.read_csv(metadata_path)
        if not metadata.empty:
            output_min = _optional_float(metadata.iloc[0].get("output_min"))
            output_max = _optional_float(metadata.iloc[0].get("output_max"))

    return QuantileMapping(table=table, knots=knots, output_min=output_min, output_max=output_max)


def _optional_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if not isinstance(value, str | int | float | np.integer | np.floating):
        raise ValueError(f"Expected optional float-compatible value, got {value!r}.")
    parsed = float(value)
    return None if np.isnan(parsed) else parsed


def _mapping_output_suffix(config: Config, mapping_dir: Path | None) -> Path:
    save_dir = Path(config.save_dir)
    if mapping_dir is None:
        return save_dir / "dynamic_range_diagnostics" / "uncalibrated"
    parts = list(mapping_dir.parts)
    if "quantile_calibration" in parts:
        idx = parts.index("quantile_calibration")
        suffix = Path(*parts[idx + 1 :])
        return save_dir / "dynamic_range_diagnostics" / suffix
    return save_dir / "dynamic_range_diagnostics" / mapping_dir.name


def _iter_stage_arrays(split_data, mapping: QuantileMapping | None):
    stages = ["uncalibrated"] + (["calibrated"] if mapping is not None else [])
    for stage in stages:
        all_pred = []
        all_target = []
        for item in split_data.hex_predictions:
            if stage == "uncalibrated":
                pred_grid = item.pred_grid
            else:
                assert mapping is not None
                pred_grid = apply_quantile_mapping(item.pred_grid, mapping)
            pred_values = pred_grid[item.valid_mask]
            target_values = item.target_values
            all_pred.append(pred_values)
            all_target.append(target_values)
            yield stage, item.hex_id, pred_values, target_values, pred_grid, item.target_grid
        yield stage, "all", np.concatenate(all_pred), np.concatenate(all_target), None, None


def _save_calibrated_plots(
    config: Config,
    split: str,
    split_data,
    mapping: QuantileMapping,
    output_dir: Path,
    robust_plot_percentile: float | None,
) -> None:
    target = get_config_target_spec(config)
    plot_dir = output_dir / split / "calibrated_plots"
    for item in split_data.hex_predictions:
        pred_grid = apply_quantile_mapping(item.pred_grid, mapping)
        hex_id = item.hex_id.removeprefix("hex").zfill(2)
        visualize_target_grids(
            gt_grid=item.target_grid,
            pred_grid=pred_grid,
            hex_id=hex_id,
            save_dir=str(plot_dir),
            target_label=target.label,
        )
        if robust_plot_percentile is not None:
            visualize_target_grids(
                gt_grid=item.target_grid,
                pred_grid=pred_grid,
                hex_id=hex_id,
                save_dir=str(plot_dir),
                target_label=target.label,
                value_percentile=robust_plot_percentile,
                diff_percentile=robust_plot_percentile,
                filename_suffix=f"_p{robust_plot_percentile:g}",
            )
        plot_hexbin_distribution(
            gt_grid=item.target_grid,
            pred_grid=pred_grid,
            hex_id=hex_id,
            save_dir=str(plot_dir),
            target_label=target.label,
            probability_scale=target.probability_scale,
        )
        plot_histogram_distribution(
            gt_grid=item.target_grid,
            pred_grid=pred_grid,
            hex_id=hex_id,
            save_dir=str(plot_dir),
            target_label=target.label,
            probability_scale=target.probability_scale,
        )
        for percentile in DEFAULT_IOU_PERCENTILES:
            pred_bin, gt_bin = get_hexel_binary_maps(pred_grid, item.target_grid, percentile=percentile)
            visualize_hexel_iou(
                item.target_grid,
                pred_grid,
                gt_bin,
                pred_bin,
                hex_id,
                str(plot_dir),
                percentile,
                target_label=target.label,
            )


def _save_threshold_exceedance_plots(
    config: Config,
    split: str,
    split_data,
    mapping: QuantileMapping | None,
    thresholds: dict[float, float],
    output_dir: Path,
    threshold_plot_scope: str,
    threshold_quantiles: tuple[float, ...],
) -> None:
    target = get_config_target_spec(config)
    scopes = ("global", "per_hex") if threshold_plot_scope == "both" else (threshold_plot_scope,)
    for item in split_data.hex_predictions:
        hex_id = item.hex_id.removeprefix("hex").zfill(2)
        stage_grids = {"uncalibrated": item.pred_grid}
        if mapping is not None:
            stage_grids["calibrated"] = apply_quantile_mapping(item.pred_grid, mapping)

        for scope in scopes:
            scope_thresholds = (
                thresholds
                if scope == "global"
                else {quantile: float(np.nanquantile(item.target_values, quantile)) for quantile in threshold_quantiles}
            )
            plot_subdir = "fixed_threshold_exceedance_plots" if scope == "per_hex" else "global_fixed_threshold_exceedance_plots"
            plot_dir = output_dir / split / plot_subdir
            for stage, pred_grid in stage_grids.items():
                for target_quantile, threshold in scope_thresholds.items():
                    visualize_fixed_threshold_exceedance(
                        target_grid=item.target_grid,
                        pred_grid=pred_grid,
                        threshold=threshold,
                        target_quantile=target_quantile,
                        hex_id=hex_id,
                        save_dir=plot_dir / stage,
                        stage=stage,
                        target_label=target.label,
                        threshold_scope=scope,
                    )


def run_dynamic_range_diagnostics(
    config: Config,
    output_dir: Path,
    splits: list[str],
    stitch_mode: str,
    center_crop_fraction: float,
    device: torch.device,
    mapping: QuantileMapping | None = None,
    save_calibrated_plots: bool = False,
    save_threshold_plots: bool = False,
    threshold_plot_scope: str = "global",
    threshold_quantiles: tuple[float, ...] = DEFAULT_THRESHOLD_QUANTILES,
    robust_plot_percentile: float | None = 99.0,
    mask_scope: str = "actual",
) -> dict[str, pd.DataFrame]:
    target = get_config_target_spec(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    dynamic_rows = []
    percentile_frames = []
    bin_summary_rows = []
    bin_detail_frames = []
    threshold_frames = []

    for split in splits:
        print(f"[dynamic-range] Collecting split={split}")
        split_data = _collect_stitched_split(
            config=config,
            split=split,
            stitch_mode=stitch_mode,
            center_crop_fraction=center_crop_fraction,
            device=device,
            mask_scope=mask_scope,
        )
        thresholds = {quantile: float(np.nanquantile(split_data.target_values, quantile)) for quantile in threshold_quantiles}

        for stage, hex_id, pred_values, target_values, _pred_grid, _target_grid in _iter_stage_arrays(split_data, mapping):
            row = {"split": split, "target": target.name, "stage": stage, "hex_id": hex_id}
            row.update(summarize_dynamic_range(pred_values, target_values))
            dynamic_rows.append(row)

            percentiles = summarize_percentiles(pred_values, target_values)
            if not percentiles.empty:
                percentiles.insert(0, "hex_id", hex_id)
                percentiles.insert(0, "stage", stage)
                percentiles.insert(0, "target", target.name)
                percentiles.insert(0, "split", split)
                percentile_frames.append(percentiles)

            threshold = summarize_threshold_exceedance(pred_values, target_values, thresholds=thresholds)
            if not threshold.empty:
                threshold.insert(0, "hex_id", hex_id)
                threshold.insert(0, "stage", stage)
                threshold.insert(0, "target", target.name)
                threshold.insert(0, "split", split)
                threshold_frames.append(threshold)

            for bin_count in DEFAULT_BIN_COUNTS:
                summary, details = summarize_bin_ranking(pred_values, target_values, bin_count=bin_count)
                summary_row = {"split": split, "target": target.name, "stage": stage, "hex_id": hex_id}
                summary_row.update(summary)
                bin_summary_rows.append(summary_row)
                if not details.empty:
                    details.insert(0, "requested_bin_count", bin_count)
                    details.insert(0, "hex_id", hex_id)
                    details.insert(0, "stage", stage)
                    details.insert(0, "target", target.name)
                    details.insert(0, "split", split)
                    bin_detail_frames.append(details)

        if save_calibrated_plots and mapping is not None:
            _save_calibrated_plots(
                config=config,
                split=split,
                split_data=split_data,
                mapping=mapping,
                output_dir=output_dir,
                robust_plot_percentile=robust_plot_percentile,
            )
        if save_threshold_plots:
            _save_threshold_exceedance_plots(
                config=config,
                split=split,
                split_data=split_data,
                mapping=mapping,
                thresholds=thresholds,
                output_dir=output_dir,
                threshold_plot_scope=threshold_plot_scope,
                threshold_quantiles=threshold_quantiles,
            )

    outputs = {
        "dynamic_range": pd.DataFrame(dynamic_rows),
        "percentiles": pd.concat(percentile_frames, ignore_index=True) if percentile_frames else pd.DataFrame(),
        "bin_ranking": pd.DataFrame(bin_summary_rows),
        "target_bin_means": pd.concat(bin_detail_frames, ignore_index=True) if bin_detail_frames else pd.DataFrame(),
        "threshold_exceedance": pd.concat(threshold_frames, ignore_index=True) if threshold_frames else pd.DataFrame(),
    }
    outputs["dynamic_range"].to_csv(output_dir / "dynamic_range_summary.csv", index=False)
    outputs["percentiles"].to_csv(output_dir / "percentile_summary.csv", index=False)
    outputs["bin_ranking"].to_csv(output_dir / "bin_ranking_summary.csv", index=False)
    outputs["target_bin_means"].to_csv(output_dir / "target_bin_means.csv", index=False)
    outputs["threshold_exceedance"].to_csv(output_dir / "threshold_exceedance.csv", index=False)
    print(f"[dynamic-range] Wrote outputs to {output_dir}")
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--mapping-dir", type=Path, default=None)
    parser.add_argument("--splits", nargs="+", default=["test"])
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--stitch-mode", default="mean", choices=["mean", "max", "center_crop", "feathered", "non_overlap"])
    parser.add_argument("--center-crop-fraction", type=float, default=0.8)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save-calibrated-plots", action="store_true")
    parser.add_argument("--save-threshold-plots", action="store_true")
    parser.add_argument("--threshold-plot-scope", default="global", choices=["global", "per_hex", "both"])
    parser.add_argument(
        "--threshold-quantiles",
        nargs="+",
        type=float,
        default=list(DEFAULT_THRESHOLD_QUANTILES),
        help="Target quantiles for fixed-threshold diagnostics. Accepts fractions like 0.95 or percents like 95.",
    )
    parser.add_argument("--robust-plot-percentile", type=float, default=99.0)
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    mapping = load_quantile_mapping(args.mapping_dir) if args.mapping_dir is not None else None
    if args.mask_scope != "actual":
        config.save_dir = get_mask_scope_save_dir(config.save_dir, args.mask_scope)
    output_dir = args.output_dir or _mapping_output_suffix(config, args.mapping_dir)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    run_dynamic_range_diagnostics(
        config=config,
        output_dir=output_dir,
        splits=args.splits,
        stitch_mode=args.stitch_mode,
        center_crop_fraction=args.center_crop_fraction,
        device=device,
        mapping=mapping,
        save_calibrated_plots=args.save_calibrated_plots,
        save_threshold_plots=args.save_threshold_plots,
        threshold_plot_scope=args.threshold_plot_scope,
        threshold_quantiles=normalize_threshold_quantiles(args.threshold_quantiles),
        robust_plot_percentile=args.robust_plot_percentile,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()
