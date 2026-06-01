"""Compare neighboring hex predictions and targets in shared buffer-overlap areas."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import TwoSlopeNorm
from rasterio.enums import Resampling
from rasterio.features import geometry_mask
from rasterio.warp import reproject

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.targets import TargetSpec, get_target_specs

DEFAULT_PAIRS = ("hex01-hex05", "hex39-hex40", "hex48-hex49", "hex16-hex47", "hex12-hex19")


@dataclass(frozen=True)
class PredictionRun:
    target: TargetSpec
    pred_dir: Path


def normalize_hex_id(hex_id: str) -> str:
    return hex_id.removeprefix("hex").zfill(2)


def parse_pair(pair: str) -> tuple[str, str]:
    parts = pair.split("-")
    if len(parts) != 2:
        raise ValueError(f"Pair must look like hex01-hex05, got {pair!r}.")
    return normalize_hex_id(parts[0]), normalize_hex_id(parts[1])


def prediction_path(pred_dir: Path, hex_id: str, target: TargetSpec) -> Path:
    candidates = [
        pred_dir / f"hexel_{hex_id}_predicted.tif",
        pred_dir / f"hexel_{hex_id}_{target.name}_predicted.tif",
        pred_dir / "predicted_hexels" / f"hexel_{hex_id}_predicted.tif",
        pred_dir / "predicted_hexels" / f"hexel_{hex_id}_{target.name}_predicted.tif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing prediction for hex{hex_id} target={target.name}. Checked: {candidates}")


def read_prediction(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        data = src.read(1, masked=True)
        profile = src.profile.copy()
    return np.asarray(np.ma.masked_invalid(data.astype("float32")).filled(np.nan), dtype=np.float32), profile


def load_target_on_prediction_grid(raw_data_dir: Path, hex_id: str, target: TargetSpec, profile: dict) -> np.ndarray:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    target_grid, _ = load_spatial_raster(
        path=getattr(paths, target.path_method)(),
        mask_path=paths.mask_grid(hex_id=hex_id, mask_scope="buffer"),
        reference_profile=profile,
    )
    return np.asarray(np.ma.masked_invalid(target_grid.astype("float32")).filled(np.nan), dtype=np.float32)


def load_mask_on_prediction_grid(raw_data_dir: Path, hex_id: str, profile: dict, mask_scope: str) -> np.ndarray:
    crs = profile.get("crs")
    transform = profile.get("transform")
    if crs is None or transform is None:
        raise ValueError("Mask outlines require prediction GeoTIFF profile with 'crs' and 'transform'.")

    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    mask_gdf = gpd.read_file(paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope))
    if mask_gdf.empty:
        raise ValueError(f"Mask contains no geometries for hex{hex_id} scope={mask_scope!r}.")
    shape = (int(profile["height"]), int(profile["width"]))
    return geometry_mask(mask_gdf.to_crs(crs).geometry, out_shape=shape, transform=transform, invert=True)


def warp_to_reference(data: np.ndarray, src_profile: dict, dst_profile: dict, resampling: Resampling) -> np.ndarray:
    nodata = -9999.0
    src = np.where(np.isfinite(data), data, nodata).astype(np.float32)
    dst = np.full((int(dst_profile["height"]), int(dst_profile["width"])), nodata, dtype=np.float32)
    reproject(
        source=src,
        destination=dst,
        src_transform=src_profile["transform"],
        src_crs=src_profile["crs"],
        src_nodata=nodata,
        dst_transform=dst_profile["transform"],
        dst_crs=dst_profile["crs"],
        dst_nodata=nodata,
        resampling=resampling,
    )
    return np.where(dst == nodata, np.nan, dst).astype(np.float32)


def pearson_corr(left: np.ndarray, right: np.ndarray) -> float:
    if left.size < 2:
        return float("nan")
    left_std = float(np.std(left))
    right_std = float(np.std(right))
    if left_std == 0.0 or right_std == 0.0:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def summarize_pair_values(left: np.ndarray, right: np.ndarray, prefix: str) -> dict[str, float]:
    valid = np.isfinite(left) & np.isfinite(right)
    left_values = left[valid].astype(np.float64, copy=False)
    right_values = right[valid].astype(np.float64, copy=False)
    if left_values.size == 0:
        return {
            f"{prefix}_valid_pixels": 0.0,
            f"{prefix}_bias": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_rmse": float("nan"),
            f"{prefix}_median_abs": float("nan"),
            f"{prefix}_p95_abs": float("nan"),
            f"{prefix}_p99_abs": float("nan"),
            f"{prefix}_max_abs": float("nan"),
            f"{prefix}_pearson": float("nan"),
            f"{prefix}_left_mean": float("nan"),
            f"{prefix}_right_mean": float("nan"),
        }
    diff = left_values - right_values
    abs_diff = np.abs(diff)
    return {
        f"{prefix}_valid_pixels": float(left_values.size),
        f"{prefix}_bias": float(np.mean(diff)),
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_median_abs": float(np.median(abs_diff)),
        f"{prefix}_p95_abs": float(np.percentile(abs_diff, 95)),
        f"{prefix}_p99_abs": float(np.percentile(abs_diff, 99)),
        f"{prefix}_max_abs": float(np.max(abs_diff)),
        f"{prefix}_pearson": pearson_corr(left_values, right_values),
        f"{prefix}_left_mean": float(np.mean(left_values)),
        f"{prefix}_right_mean": float(np.mean(right_values)),
    }


def maybe_downsample(data: np.ndarray, max_dim: int) -> np.ndarray:
    if max(data.shape) <= max_dim:
        return data
    stride = int(np.ceil(max(data.shape) / max_dim))
    return data[::stride, ::stride]


def robust_max(values: list[np.ndarray], percentile: float) -> float:
    valid_values = [arr[np.isfinite(arr)] for arr in values]
    valid_values = [arr for arr in valid_values if arr.size > 0]
    if not valid_values:
        return 1.0
    vmax = float(np.nanpercentile(np.concatenate(valid_values), percentile))
    return vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0


def robust_symmetric_limit(values: list[np.ndarray], percentile: float) -> float:
    valid_values = [np.abs(arr[np.isfinite(arr)]) for arr in values]
    valid_values = [arr for arr in valid_values if arr.size > 0]
    if not valid_values:
        return 1.0
    vmax = float(np.nanpercentile(np.concatenate(valid_values), percentile))
    return vmax if np.isfinite(vmax) and vmax > 0.0 else 1.0


def overlap_bbox(mask: np.ndarray, pad: int = 2) -> tuple[slice, slice] | None:
    rows, cols = np.where(mask)
    if rows.size == 0 or cols.size == 0:
        return None
    row_start = max(int(rows.min()) - pad, 0)
    row_stop = min(int(rows.max()) + pad + 1, mask.shape[0])
    col_start = max(int(cols.min()) - pad, 0)
    col_stop = min(int(cols.max()) + pad + 1, mask.shape[1])
    return slice(row_start, row_stop), slice(col_start, col_stop)


def finite_count(values: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(values)))


def fmt_metric(value: float, precision: int = 3) -> str:
    if not np.isfinite(value):
        return "nan"
    return f"{value:.{precision}g}"


def add_mask_outline(
    ax: plt.Axes,
    mask: np.ndarray | None,
    *,
    color: str,
    linewidth: float,
    linestyle: str = "solid",
    alpha: float = 0.9,
) -> None:
    if mask is None:
        return
    mask = np.asarray(mask, dtype=bool)
    if mask.shape[0] < 2 or mask.shape[1] < 2 or not np.any(mask) or np.all(mask):
        return
    ax.contour(mask.astype(float), levels=[0.5], colors=color, linewidths=linewidth, linestyles=linestyle, alpha=alpha)


def add_panel_outlines(ax: plt.Axes, arrays: dict[str, np.ndarray], panel_kind: str) -> None:
    if panel_kind == "left":
        add_mask_outline(ax, arrays.get("left_buffer"), color="#00D7FF", linewidth=0.8, linestyle="dashed", alpha=0.85)
        add_mask_outline(ax, arrays.get("left_actual"), color="#00D7FF", linewidth=1.1, linestyle="solid", alpha=0.95)
    elif panel_kind == "right":
        add_mask_outline(ax, arrays.get("right_buffer"), color="#FFD23F", linewidth=0.8, linestyle="dashed", alpha=0.85)
        add_mask_outline(ax, arrays.get("right_actual"), color="#FFD23F", linewidth=1.1, linestyle="solid", alpha=0.95)
    elif panel_kind == "both":
        add_panel_outlines(ax, arrays, "left")
        add_panel_outlines(ax, arrays, "right")


def plot_pair(
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    pair_label: str,
    target: TargetSpec,
    output_path: Path,
    value_percentile: float,
    diff_percentile: float,
    max_plot_dim: int,
) -> None:
    overlap = np.isfinite(left_pred) & np.isfinite(right_pred) & np.isfinite(left_target) & np.isfinite(right_target)
    arrays = [
        np.where(overlap, left_pred, np.nan),
        np.where(overlap, right_pred, np.nan),
        np.where(overlap, np.abs(left_pred - right_pred), np.nan),
        np.where(overlap, left_target, np.nan),
        np.where(overlap, right_target, np.nan),
        np.where(overlap, np.abs(left_target - right_target), np.nan),
    ]
    arrays = [maybe_downsample(arr, max_plot_dim) for arr in arrays]
    value_vmax = robust_max([arrays[0], arrays[1], arrays[3], arrays[4]], value_percentile)
    diff_vmax = robust_max([arrays[2], arrays[5]], diff_percentile)

    titles = [
        "Left prediction",
        "Right prediction warped",
        "|Prediction diff|",
        "Left target",
        "Right target warped",
        "|Target diff|",
    ]
    cmaps = ["viridis", "viridis", "magma", "viridis", "viridis", "magma"]
    vmaxes = [value_vmax, value_vmax, diff_vmax, value_vmax, value_vmax, diff_vmax]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    fig.suptitle(f"{target.label} neighbor overlap consistency: {pair_label}", fontsize=16)
    for ax, arr, title, cmap, vmax in zip(axes.flat, arrays, titles, cmaps, vmaxes, strict=True):
        im = ax.imshow(arr, origin="upper", cmap=cmap, vmin=0.0, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.75)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_overlap_row(
    left_hex: str,
    right_hex: str,
    target: TargetSpec,
    left_pred: np.ndarray,
    right_pred: np.ndarray,
    left_target: np.ndarray,
    right_target: np.ndarray,
    max_plot_dim: int,
    outline_masks: dict[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, float | str]]:
    overlap = np.isfinite(left_pred) & np.isfinite(right_pred) & np.isfinite(left_target) & np.isfinite(right_target)
    bbox = overlap_bbox(overlap)
    pair_label = f"hex{left_hex}-hex{right_hex}"
    summary: dict[str, float | str] = {
        "pair": pair_label,
        "left_hex": f"hex{left_hex}",
        "right_hex": f"hex{right_hex}",
        "target": target.name,
        "overlap_joint_pixels": float(np.count_nonzero(overlap)),
    }
    summary.update(summarize_pair_values(left_pred, right_pred, "pred"))
    summary.update(summarize_pair_values(left_target, right_target, "target"))

    if bbox is None:
        empty = np.full((1, 1), np.nan, dtype=np.float32)
        return {
            "left_target": empty,
            "right_target": empty,
            "target_diff": empty,
            "left_pred": empty,
            "right_pred": empty,
            "pred_diff": empty,
            "diff_of_diff": empty,
            "left_actual": empty.astype(bool),
            "left_buffer": empty.astype(bool),
            "right_actual": empty.astype(bool),
            "right_buffer": empty.astype(bool),
        }, summary

    target_diff = np.where(overlap, left_target - right_target, np.nan)
    pred_diff = np.where(overlap, left_pred - right_pred, np.nan)
    diff_of_diff = pred_diff - target_diff
    valid_diff_of_diff = diff_of_diff[np.isfinite(diff_of_diff)].astype(np.float64, copy=False)
    if valid_diff_of_diff.size > 0:
        summary["diff_of_diff_bias"] = float(np.mean(valid_diff_of_diff))
        summary["diff_of_diff_mae"] = float(np.mean(np.abs(valid_diff_of_diff)))
        summary["diff_of_diff_p95_abs"] = float(np.percentile(np.abs(valid_diff_of_diff), 95))
        summary["diff_of_diff_p99_abs"] = float(np.percentile(np.abs(valid_diff_of_diff), 99))
    else:
        summary["diff_of_diff_bias"] = float("nan")
        summary["diff_of_diff_mae"] = float("nan")
        summary["diff_of_diff_p95_abs"] = float("nan")
        summary["diff_of_diff_p99_abs"] = float("nan")

    masked = {
        "left_target": np.where(overlap, left_target, np.nan)[bbox],
        "right_target": np.where(overlap, right_target, np.nan)[bbox],
        "target_diff": target_diff[bbox],
        "left_pred": np.where(overlap, left_pred, np.nan)[bbox],
        "right_pred": np.where(overlap, right_pred, np.nan)[bbox],
        "pred_diff": pred_diff[bbox],
        "diff_of_diff": diff_of_diff[bbox],
    }
    if outline_masks is not None:
        for name, outline_mask in outline_masks.items():
            masked[name] = outline_mask[bbox].astype(bool, copy=False)
    return {name: maybe_downsample(values, max_plot_dim) for name, values in masked.items()}, summary


def plot_anchor_neighbor_overview(
    raw_data_dir: Path,
    anchor_hex: str,
    neighbor_hexes: list[str],
    run: PredictionRun,
    output_dir: Path,
    resampling: Resampling,
    value_percentile: float,
    diff_percentile: float,
    max_plot_dim: int,
    value_vmax: float | None = None,
) -> list[dict[str, float | str]]:
    anchor_hex = normalize_hex_id(anchor_hex)
    neighbor_hexes = [normalize_hex_id(hex_id) for hex_id in neighbor_hexes]

    anchor_pred, anchor_profile = read_prediction(prediction_path(run.pred_dir, anchor_hex, run.target))
    anchor_target = load_target_on_prediction_grid(raw_data_dir, anchor_hex, run.target, anchor_profile)
    anchor_actual_mask = load_mask_on_prediction_grid(raw_data_dir, anchor_hex, anchor_profile, "actual")
    anchor_buffer_mask = load_mask_on_prediction_grid(raw_data_dir, anchor_hex, anchor_profile, "buffer")

    row_arrays: list[dict[str, np.ndarray]] = []
    rows: list[dict[str, float | str]] = []
    for neighbor_hex in neighbor_hexes:
        neighbor_pred, neighbor_profile = read_prediction(prediction_path(run.pred_dir, neighbor_hex, run.target))
        neighbor_target = load_target_on_prediction_grid(raw_data_dir, neighbor_hex, run.target, neighbor_profile)
        neighbor_pred_warped = warp_to_reference(neighbor_pred, neighbor_profile, anchor_profile, resampling=resampling)
        neighbor_target_warped = warp_to_reference(neighbor_target, neighbor_profile, anchor_profile, resampling=resampling)
        outline_masks = {
            "left_actual": anchor_actual_mask,
            "left_buffer": anchor_buffer_mask,
            "right_actual": load_mask_on_prediction_grid(raw_data_dir, neighbor_hex, anchor_profile, "actual"),
            "right_buffer": load_mask_on_prediction_grid(raw_data_dir, neighbor_hex, anchor_profile, "buffer"),
        }
        arrays, row = build_overlap_row(
            left_hex=anchor_hex,
            right_hex=neighbor_hex,
            target=run.target,
            left_pred=anchor_pred,
            right_pred=neighbor_pred_warped,
            left_target=anchor_target,
            right_target=neighbor_target_warped,
            max_plot_dim=max_plot_dim,
            outline_masks=outline_masks,
        )
        row_arrays.append(arrays)
        rows.append(row)

    value_arrays = [arrays[name] for arrays in row_arrays for name in ("left_target", "right_target", "left_pred", "right_pred")]
    target_diff_arrays = [arrays["target_diff"] for arrays in row_arrays]
    pred_diff_arrays = [arrays["pred_diff"] for arrays in row_arrays]
    diff_of_diff_arrays = [arrays["diff_of_diff"] for arrays in row_arrays]
    resolved_value_vmax = value_vmax
    if resolved_value_vmax is None:
        resolved_value_vmax = 1.0 if run.target.name == "bp" else robust_max(value_arrays, value_percentile)
    diff_vmax = robust_symmetric_limit(target_diff_arrays + pred_diff_arrays + diff_of_diff_arrays, diff_percentile)

    n_rows = len(neighbor_hexes)
    panel_rows = n_rows * 2
    fig, axes = plt.subplots(panel_rows, 4, figsize=(17, max(2.4 * panel_rows + 1.0, 6.0)), constrained_layout=True)
    if panel_rows == 1:
        axes = np.expand_dims(axes, axis=0)
    fig.suptitle(
        f"{run.target.label} consistency in shared geographic overlap: hex{anchor_hex} vs neighbors",
        fontsize=16,
    )

    value_image = None
    diff_image = None
    diff_norm = TwoSlopeNorm(vmin=-diff_vmax, vcenter=0.0, vmax=diff_vmax)
    for row_idx, (neighbor_hex, arrays, metrics) in enumerate(zip(neighbor_hexes, row_arrays, rows, strict=True)):
        gt_row = row_idx * 2
        pred_row = gt_row + 1
        panel_specs = [
            (gt_row, 0, f"GT hex{anchor_hex}", "left_target", "value", "left"),
            (gt_row, 1, "GT neighbor", "right_target", "value", "right"),
            (gt_row, 2, f"ΔGT: hex{anchor_hex} - neighbor", "target_diff", "diff", "both"),
            (pred_row, 0, f"Pred hex{anchor_hex}", "left_pred", "value", "left"),
            (pred_row, 1, "Pred neighbor", "right_pred", "value", "right"),
            (pred_row, 2, f"ΔPred: hex{anchor_hex} - neighbor", "pred_diff", "diff", "both"),
            (pred_row, 3, "ΔΔ: ΔPred - ΔGT", "diff_of_diff", "diff", "both"),
        ]
        for panel_row, col_idx, title, array_name, scale_type, outline_kind in panel_specs:
            ax = axes[panel_row, col_idx]
            arr = arrays[array_name]
            if scale_type == "value":
                image = ax.imshow(arr, origin="upper", cmap="viridis", vmin=0.0, vmax=resolved_value_vmax)
                value_image = image
            else:
                image = ax.imshow(arr, origin="upper", cmap="coolwarm", norm=diff_norm)
                diff_image = image
            add_panel_outlines(ax, arrays, outline_kind)
            if row_idx == 0:
                ax.set_title(title)
            ax.set_xticks([])
            ax.set_yticks([])
        axes[gt_row, 0].set_ylabel(
            f"hex{anchor_hex} vs hex{neighbor_hex}\nGround truth",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
        axes[pred_row, 0].set_ylabel(
            "Prediction",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
        metric_ax = axes[gt_row, 3]
        metric_ax.axis("off")
        metric_ax.text(
            0.0,
            0.5,
            (
                f"Overlap pixels: {int(float(metrics['overlap_joint_pixels'])):,}\n"
                f"GT: MAE={fmt_metric(float(metrics['target_mae']))}, r={fmt_metric(float(metrics['target_pearson']))}\n"
                f"Pred: MAE={fmt_metric(float(metrics['pred_mae']))}, r={fmt_metric(float(metrics['pred_pearson']))}\n"
                f"ΔΔ MAE={fmt_metric(float(metrics['diff_of_diff_mae']))}\n\n"
                f"Outlines: cyan=hex{anchor_hex}, yellow=neighbor\n"
                "solid=actual, dashed=buffer"
            ),
            ha="left",
            va="center",
            fontsize=9,
        )

    if value_image is not None:
        value_axes = axes[:, [0, 1]].ravel().tolist()
        fig.colorbar(
            value_image,
            ax=value_axes,
            orientation="horizontal",
            shrink=0.75,
            pad=0.03,
            label=f"{run.target.label} value",
        )
    diff_extend = "both" if diff_percentile < 100 else "neither"
    if diff_image is not None:
        diff_axes = axes[:, 2].ravel().tolist() + axes[1::2, 3].ravel().tolist()
        fig.colorbar(
            diff_image,
            ax=diff_axes,
            orientation="horizontal",
            shrink=0.75,
            pad=0.03,
            extend=diff_extend,
            label=f"Difference scale ({run.target.label} units)",
        )

    output_path = output_dir / "plots" / f"hex{anchor_hex}_{run.target.name}_neighbor_overlap_overview.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return rows


def evaluate_pair(
    raw_data_dir: Path,
    pair: str,
    run: PredictionRun,
    output_dir: Path,
    resampling: Resampling,
    value_percentile: float,
    diff_percentile: float,
    max_plot_dim: int,
) -> dict[str, float | str]:
    left_hex, right_hex = parse_pair(pair)
    pair_label = f"hex{left_hex}-hex{right_hex}"

    left_pred, left_profile = read_prediction(prediction_path(run.pred_dir, left_hex, run.target))
    right_pred, right_profile = read_prediction(prediction_path(run.pred_dir, right_hex, run.target))
    left_target = load_target_on_prediction_grid(raw_data_dir, left_hex, run.target, left_profile)
    right_target = load_target_on_prediction_grid(raw_data_dir, right_hex, run.target, right_profile)

    right_pred_warped = warp_to_reference(right_pred, right_profile, left_profile, resampling=resampling)
    right_target_warped = warp_to_reference(right_target, right_profile, left_profile, resampling=resampling)

    pred_valid = np.isfinite(left_pred) & np.isfinite(right_pred_warped)
    target_valid = np.isfinite(left_target) & np.isfinite(right_target_warped)
    both_valid = pred_valid & target_valid
    row: dict[str, float | str] = {
        "pair": pair_label,
        "left_hex": f"hex{left_hex}",
        "right_hex": f"hex{right_hex}",
        "target": run.target.name,
        "overlap_pred_pixels": float(np.count_nonzero(pred_valid)),
        "overlap_target_pixels": float(np.count_nonzero(target_valid)),
        "overlap_joint_pixels": float(np.count_nonzero(both_valid)),
    }
    row.update(summarize_pair_values(left_pred, right_pred_warped, "pred"))
    row.update(summarize_pair_values(left_target, right_target_warped, "target"))
    if np.count_nonzero(both_valid) > 0:
        pred_abs = np.abs(left_pred[both_valid] - right_pred_warped[both_valid])
        target_abs = np.abs(left_target[both_valid] - right_target_warped[both_valid])
        row["pred_to_target_mae_ratio"] = float(np.mean(pred_abs) / np.mean(target_abs)) if np.mean(target_abs) > 0.0 else float("nan")
        row["pred_minus_target_abs_mae"] = float(np.mean(pred_abs - target_abs))
    else:
        row["pred_to_target_mae_ratio"] = float("nan")
        row["pred_minus_target_abs_mae"] = float("nan")

    plot_pair(
        left_pred=left_pred,
        right_pred=right_pred_warped,
        left_target=left_target,
        right_target=right_target_warped,
        pair_label=pair_label,
        target=run.target,
        output_path=output_dir / "plots" / f"{pair_label}_{run.target.name}_neighbor_overlap.png",
        value_percentile=value_percentile,
        diff_percentile=diff_percentile,
        max_plot_dim=max_plot_dim,
    )
    return row


def parse_prediction_runs(args: argparse.Namespace) -> list[PredictionRun]:
    runs = []
    for target_name, pred_dir in (("bp", args.bp_pred_dir), ("fi", args.fi_pred_dir), ("ros", args.ros_pred_dir)):
        if pred_dir is not None:
            runs.append(PredictionRun(target=get_target_specs(target_name)[0], pred_dir=pred_dir))
    if not runs:
        raise ValueError("At least one prediction directory must be provided.")
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS), help="Neighbor pairs like hex01-hex05.")
    parser.add_argument("--bp-pred-dir", type=Path, default=None)
    parser.add_argument("--fi-pred-dir", type=Path, default=None)
    parser.add_argument("--ros-pred-dir", type=Path, default=None)
    parser.add_argument("--resampling", choices=["nearest", "bilinear"], default="nearest")
    parser.add_argument("--value-percentile", type=float, default=99.0)
    parser.add_argument("--diff-percentile", type=float, default=99.0)
    parser.add_argument("--max-plot-dim", type=int, default=1800)
    parser.add_argument("--overview-anchor-hex", default=None, help="Anchor hex for a multi-neighbor overview plot, e.g. hex01.")
    parser.add_argument("--overview-neighbors", nargs="+", default=None, help="Neighbor hexes for the overview plot.")
    parser.add_argument("--overview-only", action="store_true", help="Skip individual pair plots and only write the overview plot.")
    parser.add_argument("--overview-value-vmax", type=float, default=None, help="Fixed value colorbar max for overview plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    resampling = Resampling.nearest if args.resampling == "nearest" else Resampling.bilinear
    runs = parse_prediction_runs(args)
    rows = []
    overview_rows = []
    if not args.overview_only:
        for run in runs:
            for pair in args.pairs:
                rows.append(
                    evaluate_pair(
                        raw_data_dir=args.raw_data_dir,
                        pair=pair,
                        run=run,
                        output_dir=args.output_dir,
                        resampling=resampling,
                        value_percentile=args.value_percentile,
                        diff_percentile=args.diff_percentile,
                        max_plot_dim=args.max_plot_dim,
                    )
                )
    if args.overview_neighbors is not None:
        if args.overview_anchor_hex is None:
            raise ValueError("--overview-anchor-hex is required when --overview-neighbors is provided.")
        for run in runs:
            overview_rows.extend(
                plot_anchor_neighbor_overview(
                    raw_data_dir=args.raw_data_dir,
                    anchor_hex=args.overview_anchor_hex,
                    neighbor_hexes=args.overview_neighbors,
                    run=run,
                    output_dir=args.output_dir,
                    resampling=resampling,
                    value_percentile=args.value_percentile,
                    diff_percentile=args.diff_percentile,
                    max_plot_dim=args.max_plot_dim,
                    value_vmax=args.overview_value_vmax,
                )
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if rows:
        summary = pd.DataFrame(rows)
        summary_path = args.output_dir / "neighbor_overlap_consistency.csv"
        summary.to_csv(summary_path, index=False)
        print(summary.to_string(index=False))
        print(f"Wrote {summary_path}")
    if overview_rows:
        overview_summary = pd.DataFrame(overview_rows)
        overview_summary_path = args.output_dir / "anchor_neighbor_overlap_summary.csv"
        overview_summary.to_csv(overview_summary_path, index=False)
        print(overview_summary.to_string(index=False))
        print(f"Wrote {overview_summary_path}")


if __name__ == "__main__":
    main()
