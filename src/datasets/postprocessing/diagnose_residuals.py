import argparse
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import rasterio

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.spatial.fuel import load_fuel_grid
from data_preparation.spatial.utils import load_spatial_raster
from src.config import Config
from src.datasets.postprocessing.utils import (
    as_float_array_with_nan,
    get_config_target_spec,
    get_mask_scope_save_dir,
    load_target_grid_for_mask_scope,
    validate_patch_metadata_mask_scope,
)
from src.train import load_config

EDGE_DISTANCE_BINS = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, np.inf], dtype=np.float32)
EDGE_DISTANCE_LABELS = {
    0: "0",
    1: "1",
    2: "2-3",
    3: "4-7",
    4: "8-15",
    5: "16-31",
    6: "32-63",
    7: "64-127",
    8: "128+",
}
SPATIAL_CORRELATION_LAGS = (1, 2, 4, 8, 16, 32, 64, 128, 256)


def _normalize_hex_id(hex_id: object) -> str:
    if isinstance(hex_id, int | np.integer):
        return f"{int(hex_id):02d}"
    if isinstance(hex_id, float | np.floating) and np.isfinite(hex_id) and float(hex_id).is_integer():
        return f"{int(hex_id):02d}"
    return str(hex_id).zfill(2)


def _resolve_split_name(config: Config, split: str) -> str:
    if split == "train":
        return config.data.train_split
    if split == "val":
        return config.data.val_split
    if split == "test":
        return config.data.test_split
    return split


def _prediction_path(config: Config, hex_id: str, mask_scope: str = "actual") -> Path:
    return Path(get_mask_scope_save_dir(config.save_dir, mask_scope)) / "predicted_hexels" / f"hexel_{hex_id}_predicted.tif"


def _diagnostic_array_path(
    config: Config,
    split: str,
    hex_id: str,
    filename: str,
    patch_diagnostics_dir: str | None = None,
    mask_scope: str = "actual",
) -> Path:
    candidates = []
    if patch_diagnostics_dir is not None:
        candidates.append(Path(patch_diagnostics_dir) / f"hex{hex_id}" / filename)

    candidates.extend(
        [
            Path(get_mask_scope_save_dir(config.save_dir, mask_scope)) / "patch_stitch_diagnostics" / f"hex{hex_id}" / filename,
            Path(get_mask_scope_save_dir(config.save_dir, mask_scope)) / "patch_stitch_diagnostics" / split / f"hex{hex_id}" / filename,
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate

    joined = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"Missing patch/stitch diagnostic array {filename!r} for hex{hex_id}. Run diagnose_patch_stitching first. Checked:\n  {joined}"
    )


def _read_predicted_grid(path: Path) -> tuple[np.ndarray, dict]:
    if not path.exists():
        raise FileNotFoundError(f"Predicted hexel not found: {path}")

    with rasterio.open(path) as src:
        pred = src.read(1, masked=True)
        profile = src.profile.copy()

    return as_float_array_with_nan(pred), profile


def _error_summary(diff_values: np.ndarray) -> dict[str, float]:
    diff_values = diff_values[np.isfinite(diff_values)]
    if diff_values.size == 0:
        return {
            "count": 0.0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "bias": float("nan"),
            "median_diff": float("nan"),
            "p95_abs_error": float("nan"),
            "p99_abs_error": float("nan"),
            "max_abs_error": float("nan"),
        }

    abs_error = np.abs(diff_values)
    return {
        "count": float(diff_values.size),
        "mae": float(np.mean(abs_error)),
        "rmse": float(np.sqrt(np.mean(diff_values**2))),
        "bias": float(np.mean(diff_values)),
        "median_diff": float(np.median(diff_values)),
        "p95_abs_error": float(np.percentile(abs_error, 95)),
        "p99_abs_error": float(np.percentile(abs_error, 99)),
        "max_abs_error": float(np.max(abs_error)),
    }


def summarize_grouped_errors(
    diff: np.ndarray,
    valid_mask: np.ndarray,
    group_values: np.ndarray,
    group_name: str,
    target: str,
    hex_id: str,
    label_lookup: Mapping[Any, str] | None = None,
    max_groups: int | None = None,
) -> pd.DataFrame:
    flat_valid = valid_mask.ravel()
    flat_diff = diff.ravel()[flat_valid]
    flat_groups = group_values.ravel()[flat_valid]

    finite = np.isfinite(flat_diff) & np.isfinite(flat_groups)
    flat_diff = flat_diff[finite]
    flat_groups = flat_groups[finite]
    if flat_diff.size == 0:
        return pd.DataFrame()

    unique_values, counts = np.unique(flat_groups, return_counts=True)
    if max_groups is not None and unique_values.size > max_groups:
        selected = unique_values[np.argsort(counts)[::-1][:max_groups]]
    else:
        selected = unique_values

    rows = []
    for value in selected:
        group_diff = flat_diff[flat_groups == value]
        if group_diff.size == 0:
            continue
        label = label_lookup.get(value, str(value)) if label_lookup is not None else str(value)
        row: dict[str, object] = {
            "target": target,
            "hex_id": hex_id,
            "group": group_name,
            "group_value": label,
        }
        row.update(_error_summary(group_diff))
        rows.append(row)

    return pd.DataFrame(rows)


def _edge_distance_codes(edge_distance: np.ndarray) -> np.ndarray:
    codes = np.full(edge_distance.shape, -1, dtype=np.int16)
    finite = np.isfinite(edge_distance)
    codes[finite] = np.digitize(edge_distance[finite], EDGE_DISTANCE_BINS[1:-1], right=False).astype(np.int16)
    return codes


def _coverage_codes(coverage: np.ndarray) -> tuple[np.ndarray, dict[int, str]]:
    codes = np.full(coverage.shape, -1, dtype=np.int16)
    finite = np.isfinite(coverage) & (coverage > 0)
    codes[finite] = np.minimum(coverage[finite], 4).astype(np.int16)
    return codes, {1: "1", 2: "2", 3: "3", 4: "4+"}


def _elevation_quantile_codes(elevation: np.ndarray, valid_mask: np.ndarray, n_bins: int = 5) -> tuple[np.ndarray, dict[int, str]]:
    codes = np.full(elevation.shape, -1, dtype=np.int16)
    valid_values = elevation[valid_mask & np.isfinite(elevation)]
    if valid_values.size == 0:
        return codes, {}

    edges = np.unique(np.nanquantile(valid_values, np.linspace(0.0, 1.0, n_bins + 1)))
    if edges.size < 2:
        codes[valid_mask & np.isfinite(elevation)] = 0
        return codes, {0: f"{float(edges[0]):.3g}"}

    finite = np.isfinite(elevation)
    codes[finite] = np.digitize(elevation[finite], edges[1:-1], right=True).astype(np.int16)
    labels = {i: f"q{i + 1}:{edges[i]:.3g}-{edges[i + 1]:.3g}" for i in range(edges.size - 1)}
    return codes, labels


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    finite = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(finite) < 2:
        return float("nan")
    x_valid = x[finite]
    y_valid = y[finite]
    if np.std(x_valid) == 0.0 or np.std(y_valid) == 0.0:
        return float("nan")
    return float(np.corrcoef(x_valid, y_valid)[0, 1])


def _masked_correlation(x: np.ndarray, y: np.ndarray, valid_mask: np.ndarray) -> tuple[float, int]:
    valid = valid_mask & np.isfinite(x) & np.isfinite(y)
    n = int(np.count_nonzero(valid))
    if n < 2:
        return float("nan"), n

    x_arr = x.astype(np.float64, copy=False)
    y_arr = y.astype(np.float64, copy=False)
    sum_x = np.sum(x_arr, where=valid, dtype=np.float64)
    sum_y = np.sum(y_arr, where=valid, dtype=np.float64)
    sum_xx = np.sum(x_arr * x_arr, where=valid, dtype=np.float64)
    sum_yy = np.sum(y_arr * y_arr, where=valid, dtype=np.float64)
    sum_xy = np.sum(x_arr * y_arr, where=valid, dtype=np.float64)

    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_xx - sum_x**2
    denom_y = n * sum_yy - sum_y**2
    denominator = np.sqrt(denom_x * denom_y)
    if denominator <= 0.0 or not np.isfinite(denominator):
        return float("nan"), n
    return float(numerator / denominator), n


def lagged_spatial_correlations(
    values: np.ndarray,
    valid_mask: np.ndarray,
    lags: tuple[int, ...] = SPATIAL_CORRELATION_LAGS,
) -> pd.DataFrame:
    rows = []
    for lag in lags:
        if lag >= values.shape[0] and lag >= values.shape[1]:
            continue

        if lag < values.shape[0]:
            corr, count = _masked_correlation(
                values[:-lag, :],
                values[lag:, :],
                valid_mask[:-lag, :] & valid_mask[lag:, :],
            )
            rows.append({"lag_px": lag, "axis": "row", "correlation": corr, "pair_count": float(count)})

        if lag < values.shape[1]:
            corr, count = _masked_correlation(
                values[:, :-lag],
                values[:, lag:],
                valid_mask[:, :-lag] & valid_mask[:, lag:],
            )
            rows.append({"lag_px": lag, "axis": "col", "correlation": corr, "pair_count": float(count)})

    return pd.DataFrame(rows)


def _load_context_grids(
    raw_data_dir: str, hex_id: str, reference_profile: dict, mask_scope: str = "actual"
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    fuel = as_float_array_with_nan(
        load_fuel_grid(root_dir=raw_data_dir, hex_id=hex_id, reference_profile=reference_profile, mask_scope=mask_scope)
    )
    firezone, _ = load_spatial_raster(
        path=paths.firezones_grid(hex_id=hex_id),
        mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
        reference_profile=reference_profile,
    )
    elevation, _ = load_spatial_raster(
        path=paths.elevation_grid(hex_id=hex_id),
        mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
        reference_profile=reference_profile,
    )
    return fuel, as_float_array_with_nan(firezone), as_float_array_with_nan(elevation)


def run_residual_diagnostics(
    config: Config,
    split: str = "test",
    hex_ids: list[str] | None = None,
    save_dir: str | None = None,
    patch_diagnostics_dir: str | None = None,
    max_fire_zones: int | None = None,
    mask_scope: str = "actual",
) -> dict[str, pd.DataFrame]:
    target = get_config_target_spec(config)
    split_name = _resolve_split_name(config, split)
    split_df = pd.read_csv(os.path.join(config.data.root_dir, split_name))
    split_df = split_df[split_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)
    validate_patch_metadata_mask_scope(split_df, mask_scope)
    split_df["hex_id_norm"] = split_df["hex_id"].map(_normalize_hex_id)

    requested_hex_ids = {_normalize_hex_id(hex_id) for hex_id in hex_ids} if hex_ids else set(split_df["hex_id_norm"].unique())
    scope_save_dir = Path(get_mask_scope_save_dir(config.save_dir, mask_scope))
    out_dir = Path(save_dir or scope_save_dir / "residual_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)

    hex_rows = []
    grouped_rows = []
    spatial_corr_rows = []

    for hex_id in sorted(requested_hex_ids):
        if split_df[split_df["hex_id_norm"] == hex_id].empty:
            print(f"[residual diagnostics] Skipping hex{hex_id}: no windows in split {split_name}.")
            continue

        pred_grid, pred_profile = _read_predicted_grid(_prediction_path(config, hex_id, mask_scope=mask_scope))
        paths = Paths(hex_id=hex_id, root_dir=config.data.raw_data_dir)
        target_grid, pred_grid = load_target_grid_for_mask_scope(
            paths=paths,
            target=target,
            pred_grid=pred_grid,
            profile=pred_profile,
            mask_scope=mask_scope,
            hex_id=hex_id,
        )
        target_grid = as_float_array_with_nan(target_grid)

        valid_mask = np.isfinite(pred_grid) & np.isfinite(target_grid)
        diff = pred_grid - target_grid
        abs_error = np.abs(diff)

        edge_distance = np.load(
            _diagnostic_array_path(config, split, hex_id, "mean_edge_distance.npy", patch_diagnostics_dir, mask_scope=mask_scope)
        )
        coverage = np.load(
            _diagnostic_array_path(config, split, hex_id, "coverage_count.npy", patch_diagnostics_dir, mask_scope=mask_scope)
        )
        fuel, firezone, elevation = _load_context_grids(config.data.raw_data_dir, hex_id, pred_profile, mask_scope=mask_scope)

        hex_row: dict[str, object] = {
            "target": target.name,
            "hex_id": hex_id,
            "split": split_name,
            "mask_scope": mask_scope,
        }
        hex_row.update(_error_summary(diff[valid_mask]))
        hex_row["corr_abs_error_edge_distance"] = _correlation(abs_error[valid_mask], edge_distance[valid_mask])
        hex_row["corr_abs_error_elevation"] = _correlation(abs_error[valid_mask], elevation[valid_mask])
        hex_rows.append(hex_row)

        for quantity, values in [("residual", diff), ("abs_error", abs_error)]:
            spatial_corr = lagged_spatial_correlations(values=values, valid_mask=valid_mask)
            if not spatial_corr.empty:
                spatial_corr.insert(0, "quantity", quantity)
                spatial_corr.insert(0, "hex_id", hex_id)
                spatial_corr.insert(0, "target", target.name)
                spatial_corr_rows.append(spatial_corr)

        edge_codes = _edge_distance_codes(edge_distance)
        grouped_rows.append(
            summarize_grouped_errors(
                diff=diff,
                valid_mask=valid_mask & (edge_codes >= 0),
                group_values=edge_codes,
                group_name="edge_distance_px",
                target=target.name,
                hex_id=hex_id,
                label_lookup=EDGE_DISTANCE_LABELS,
            )
        )

        coverage_codes, coverage_labels = _coverage_codes(coverage)
        grouped_rows.append(
            summarize_grouped_errors(
                diff=diff,
                valid_mask=valid_mask & (coverage_codes >= 0),
                group_values=coverage_codes,
                group_name="coverage_count",
                target=target.name,
                hex_id=hex_id,
                label_lookup=coverage_labels,
            )
        )

        grouped_rows.append(
            summarize_grouped_errors(
                diff=diff,
                valid_mask=valid_mask & np.isfinite(fuel) & (fuel >= 0),
                group_values=fuel,
                group_name="fuel_group",
                target=target.name,
                hex_id=hex_id,
            )
        )

        grouped_rows.append(
            summarize_grouped_errors(
                diff=diff,
                valid_mask=valid_mask & np.isfinite(firezone) & (firezone > 0),
                group_values=firezone,
                group_name="fire_zone",
                target=target.name,
                hex_id=hex_id,
                max_groups=max_fire_zones,
            )
        )

        elevation_codes, elevation_labels = _elevation_quantile_codes(elevation, valid_mask=valid_mask)
        grouped_rows.append(
            summarize_grouped_errors(
                diff=diff,
                valid_mask=valid_mask & (elevation_codes >= 0),
                group_values=elevation_codes,
                group_name="elevation_quantile",
                target=target.name,
                hex_id=hex_id,
                label_lookup=elevation_labels,
            )
        )

    hex_summary = pd.DataFrame(hex_rows)
    grouped_summary = pd.concat([df for df in grouped_rows if not df.empty], ignore_index=True) if grouped_rows else pd.DataFrame()
    spatial_corr_summary = (
        pd.concat([df for df in spatial_corr_rows if not df.empty], ignore_index=True) if spatial_corr_rows else pd.DataFrame()
    )

    hex_summary_path = out_dir / "residual_hex_summary.csv"
    grouped_summary_path = out_dir / "residual_grouped_summary.csv"
    spatial_corr_summary_path = out_dir / "residual_spatial_correlations.csv"
    hex_summary.to_csv(hex_summary_path, index=False)
    grouped_summary.to_csv(grouped_summary_path, index=False)
    spatial_corr_summary.to_csv(spatial_corr_summary_path, index=False)
    print(f"[residual diagnostics] Wrote hex summary to {hex_summary_path}")
    print(f"[residual diagnostics] Wrote grouped summary to {grouped_summary_path}")
    print(f"[residual diagnostics] Wrote spatial correlations to {spatial_corr_summary_path}")

    return {"hex": hex_summary, "grouped": grouped_summary, "spatial_correlations": spatial_corr_summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run residual diagnostics for predicted stitched hexels.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML config.")
    parser.add_argument("--split", default="test", help="Split alias train/val/test or a split CSV filename.")
    parser.add_argument("--hex-id", nargs="*", default=None, help="Optional hex IDs to diagnose.")
    parser.add_argument("--save-dir", default=None, help="Optional output directory for residual CSVs.")
    parser.add_argument("--patch-diagnostics-dir", default=None, help="Optional directory produced by diagnose_patch_stitching.")
    parser.add_argument("--max-fire-zones", type=int, default=None, help="Optionally restrict fire-zone summaries to largest N zones.")
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    run_residual_diagnostics(
        config=config,
        split=args.split,
        hex_ids=args.hex_id,
        save_dir=args.save_dir,
        patch_diagnostics_dir=args.patch_diagnostics_dir,
        max_fire_zones=args.max_fire_zones,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()
