"""Orchestration over paired BP/FI stitched hexels and hazard artifact writing."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

import numpy as np
import rasterio
from rasterio.profiles import Profile

from src.config import DEFAULT_HAZARD_BIN_THRESHOLDS
from src.datasets.postprocessing.hazard import bin_scaled_hazard, compute_raw_hazard, scale_hazard
from src.datasets.postprocessing.hazard_metrics import calculate_hazard_class_metrics
from src.datasets.postprocessing.hexel_reconstruction import StitchedHexel
from src.datasets.postprocessing.visualize_predictions import visualize_target_grids


@dataclass(frozen=True)
class HazardHexelResult:
    hex_id: str
    pred_raw_hazard: np.ndarray
    pred_scaled_hazard: np.ndarray
    pred_binned_hazard: np.ndarray
    gt_raw_hazard: np.ndarray
    gt_scaled_hazard: np.ndarray
    gt_binned_hazard: np.ndarray
    profile: Profile
    metrics: dict[str, float | np.ndarray]
    actual_support_mask: np.ndarray | None = None
    buffer_support_mask: np.ndarray | None = None


def pair_stitched_hexels(
    bp_hexels: Iterable[StitchedHexel],
    fi_hexels: Iterable[StitchedHexel],
) -> Iterator[tuple[StitchedHexel, StitchedHexel]]:
    """Yield aligned (bp, fi) hexels, validating identical hex-id sequences."""
    bp_iter = iter(bp_hexels)
    fi_iter = iter(fi_hexels)
    index = 0
    while True:
        try:
            bp_hexel = next(bp_iter)
            bp_done = False
        except StopIteration:
            bp_done = True
        try:
            fi_hexel = next(fi_iter)
            fi_done = False
        except StopIteration:
            fi_done = True

        if bp_done and fi_done:
            return
        index += 1
        if bp_done or fi_done:
            raise ValueError(f"BP/FI hexel counts differ before pair {index}")
        if bp_hexel.hex_id != fi_hexel.hex_id:
            raise ValueError(f"BP/FI hexel sequence mismatch: {bp_hexel.hex_id} vs {fi_hexel.hex_id}")
        yield bp_hexel, fi_hexel


def compute_hazard_hexel(
    bp_hexel: StitchedHexel,
    fi_hexel: StitchedHexel,
    *,
    denominator: float,
    pred_denominator: float | None = None,
    fi_cap: float | None = 10000.0,
    scale_to: float = 100.0,
    bin_thresholds: list[float] | None = None,
    invalid_class: int = 0,
) -> HazardHexelResult:
    """Build a hazard result from one paired BP/FI stitched hexel."""
    if bp_hexel.hex_id != fi_hexel.hex_id:
        raise ValueError(f"hex_id mismatch: {bp_hexel.hex_id!r} vs {fi_hexel.hex_id!r}")
    if bp_hexel.target.name != "bp":
        raise ValueError(f"bp_hexel must carry the 'bp' target, got {bp_hexel.target.name!r}")
    if fi_hexel.target.name != "fi":
        raise ValueError(f"fi_hexel must carry the 'fi' target, got {fi_hexel.target.name!r}")

    shapes = {
        bp_hexel.pred_grid.shape,
        bp_hexel.gt_grid.shape,
        fi_hexel.pred_grid.shape,
        fi_hexel.gt_grid.shape,
    }
    if len(shapes) != 1:
        raise ValueError(f"BP/FI predicted and GT grids must share a shape, got {shapes}")

    for key in ("crs", "transform"):
        bp_value = bp_hexel.profile.get(key)
        fi_value = fi_hexel.profile.get(key)
        if bp_value is not None and fi_value is not None and bp_value != fi_value:
            raise ValueError(f"BP/FI profiles must share {key}, got {bp_value!r} vs {fi_value!r}")

    if bin_thresholds is None:
        bin_thresholds = list(DEFAULT_HAZARD_BIN_THRESHOLDS)

    pred_raw = compute_raw_hazard(bp_hexel.pred_grid, fi_hexel.pred_grid, fi_cap)
    gt_raw = compute_raw_hazard(bp_hexel.gt_grid, fi_hexel.gt_grid, fi_cap)
    pred_scaled = scale_hazard(pred_raw, pred_denominator if pred_denominator is not None else denominator, scale_to)
    gt_scaled = scale_hazard(gt_raw, denominator, scale_to)
    pred_binned = bin_scaled_hazard(pred_scaled, bin_thresholds, invalid_class)
    gt_binned = bin_scaled_hazard(gt_scaled, bin_thresholds, invalid_class)

    metrics = calculate_hazard_class_metrics(
        pred_binned,
        gt_binned,
        invalid_class=invalid_class,
        num_classes=len(bin_thresholds) + 1,
    )

    return HazardHexelResult(
        hex_id=bp_hexel.hex_id,
        pred_raw_hazard=pred_raw,
        pred_scaled_hazard=pred_scaled,
        pred_binned_hazard=pred_binned,
        gt_raw_hazard=gt_raw,
        gt_scaled_hazard=gt_scaled,
        gt_binned_hazard=gt_binned,
        profile=bp_hexel.profile,
        metrics=metrics,
        actual_support_mask=bp_hexel.actual_support_mask,
        buffer_support_mask=bp_hexel.buffer_support_mask,
    )


def _write_hazard_raster(array: np.ndarray, profile: Profile, out_path: str, *, dtype: str, nodata: float) -> None:
    prof = dict(profile)
    prof.update(count=1, dtype=dtype, nodata=nodata)
    write_array = np.where(np.isfinite(array), array, nodata).astype(dtype)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with rasterio.open(out_path, "w", **prof) as dst:
        dst.write(write_array, 1)


def save_hazard_hexel_artifacts(
    result: HazardHexelResult,
    save_dir: str,
    *,
    save_plots: bool = True,
    invalid_class: int = 0,
) -> list[str]:
    """Write hazard GeoTIFFs (and optional plots) under a hazard-specific subdir."""
    base = os.path.join(save_dir, "hazard_hexels")
    products: dict[tuple[str, str], np.ndarray] = {
        ("predicted", "raw"): result.pred_raw_hazard,
        ("predicted", "scaled"): result.pred_scaled_hazard,
        ("predicted", "binned"): result.pred_binned_hazard,
        ("ground_truth", "raw"): result.gt_raw_hazard,
        ("ground_truth", "scaled"): result.gt_scaled_hazard,
        ("ground_truth", "binned"): result.gt_binned_hazard,
    }

    written: list[str] = []
    for (kind, product), array in products.items():
        out_path = os.path.join(base, product, f"hexel_{result.hex_id}_{kind}_{product}_hazard.tif")
        if product == "binned":
            _write_hazard_raster(array, result.profile, out_path, dtype="int32", nodata=invalid_class)
        else:
            _write_hazard_raster(array, result.profile, out_path, dtype="float32", nodata=-9999.0)
        written.append(out_path)

    if save_plots:
        for product, gt_grid, pred_grid in (
            ("raw", result.gt_raw_hazard, result.pred_raw_hazard),
            ("scaled", result.gt_scaled_hazard, result.pred_scaled_hazard),
        ):
            visualize_target_grids(
                gt_grid=gt_grid,
                pred_grid=pred_grid,
                hex_id=result.hex_id,
                save_dir=base,
                target_label=f"{product.capitalize()} Hazard",
                target_name="hazard",
                filename_suffix=f"_{product}",
                actual_support_mask=result.actual_support_mask,
                buffer_support_mask=result.buffer_support_mask,
            )

    return written
