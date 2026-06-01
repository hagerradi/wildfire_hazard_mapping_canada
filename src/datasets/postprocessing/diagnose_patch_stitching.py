import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.config import Config
from src.datasets.postprocessing.stitch_hexel import stitch_windows_with_diagnostics
from src.datasets.postprocessing.utils import (
    as_float_array_with_nan,
    get_config_target_spec,
    get_mask_scope_save_dir,
    get_target_channel_index,
    load_target_grid_for_mask_scope,
    validate_patch_metadata_mask_scope,
)
from src.train import load_config


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


def _load_target_windows(
    data_dir: str,
    hex_df: pd.DataFrame,
    target_channel_index: int,
) -> tuple[list[np.ndarray], list[tuple[int, int]], list[np.ndarray]]:
    windows = []
    coords = []
    masks = []

    for row in hex_df.itertuples(index=False):
        patch_path = os.path.join(data_dir, row.filename)
        patch = np.load(patch_path, mmap_mode="r")
        if target_channel_index >= patch.shape[2]:
            raise ValueError(f"target_channel_index={target_channel_index} is out of bounds for patch {patch_path}: {patch.shape}")

        target_window = np.asarray(patch[:, :, target_channel_index], dtype=np.float32)
        windows.append(target_window)
        masks.append(np.isfinite(target_window))
        coords.append((int(row.row), int(row.col)))

    return windows, coords, masks


def _target_reconstruction_stats(reconstructed: np.ndarray, target_grid: np.ndarray) -> dict[str, float]:
    target_arr = as_float_array_with_nan(target_grid)
    valid_target = np.isfinite(target_arr)
    valid_both = valid_target & np.isfinite(reconstructed)
    missing_after_stitch = valid_target & ~np.isfinite(reconstructed)

    if np.any(valid_both):
        diff = reconstructed[valid_both] - target_arr[valid_both]
        mae = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))
        exact_fraction = float(np.mean(diff == 0.0))
    else:
        mae = float("nan")
        max_abs = float("nan")
        exact_fraction = float("nan")

    return {
        "target_valid_pixel_count": float(np.count_nonzero(valid_target)),
        "reconstructed_valid_pixel_count": float(np.count_nonzero(np.isfinite(reconstructed))),
        "valid_target_missing_after_stitch_count": float(np.count_nonzero(missing_after_stitch)),
        "reconstruction_mae": mae,
        "reconstruction_max_abs_error": max_abs,
        "reconstruction_exact_fraction": exact_fraction,
    }


def run_patch_stitch_diagnostics(
    config: Config,
    split: str = "test",
    hex_ids: list[str] | None = None,
    save_dir: str | None = None,
    mask_scope: str = "actual",
) -> pd.DataFrame:
    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    target = get_config_target_spec(config)
    target_channel_index = get_target_channel_index(
        data_dir=data_dir,
        modelling_approach=config.modelling_approach,
        target=target,
    )

    split_name = _resolve_split_name(config, split)
    split_df = pd.read_csv(os.path.join(data_dir, split_name))
    split_df = split_df[split_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)
    validate_patch_metadata_mask_scope(split_df, mask_scope)
    split_df["hex_id_norm"] = split_df["hex_id"].map(_normalize_hex_id)

    requested_hex_ids = {_normalize_hex_id(hex_id) for hex_id in hex_ids} if hex_ids else set(split_df["hex_id_norm"].unique())
    scope_save_dir = get_mask_scope_save_dir(config.save_dir, mask_scope)
    out_dir = save_dir or os.path.join(scope_save_dir, "patch_stitch_diagnostics", split)
    os.makedirs(out_dir, exist_ok=True)

    rows = []
    for hex_id in sorted(requested_hex_ids):
        hex_df = split_df[split_df["hex_id_norm"] == hex_id]
        if hex_df.empty:
            print(f"[patch diagnostics] Skipping hex{hex_id}: no windows in split {split_name}.")
            continue

        all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
        gt_elevation_grid, gt_elevation_profile = load_spatial_raster(
            path=all_paths.elevation_grid(hex_id=hex_id),
            mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
        )

        windows, coords, masks = _load_target_windows(
            data_dir=data_dir,
            hex_df=hex_df,
            target_channel_index=target_channel_index,
        )
        reconstructed, diagnostics = stitch_windows_with_diagnostics(
            windows=windows,
            coords=coords,
            masks=masks,
            original_shape=tuple(gt_elevation_grid.shape),
            mode="mean",
        )
        target_grid, reconstructed = load_target_grid_for_mask_scope(
            paths=all_paths,
            target=target,
            pred_grid=reconstructed,
            profile=gt_elevation_profile,
            mask_scope=mask_scope,
            hex_id=hex_id,
        )

        hex_out_dir = os.path.join(out_dir, f"hex{hex_id}")
        os.makedirs(hex_out_dir, exist_ok=True)
        np.save(os.path.join(hex_out_dir, "target_reconstruction.npy"), reconstructed.astype(np.float32))
        np.save(os.path.join(hex_out_dir, "coverage_count.npy"), diagnostics.coverage_count.astype(np.int32))
        np.save(os.path.join(hex_out_dir, "overlap_range.npy"), diagnostics.overlap_range.astype(np.float32))
        np.save(os.path.join(hex_out_dir, "mean_edge_distance.npy"), diagnostics.mean_edge_distance.astype(np.float32))

        row = {
            "hex_id": hex_id,
            "target": target.name,
            "split": split_name,
            "mask_scope": mask_scope,
            "window_count": float(len(hex_df)),
        }
        row.update(diagnostics.summary)
        row.update(_target_reconstruction_stats(reconstructed=reconstructed, target_grid=target_grid))
        rows.append(row)

    summary_df = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, "patch_stitch_diagnostics.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[patch diagnostics] Wrote summary to {summary_path}")
    return summary_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run patch extraction/stitching diagnostics for a config split.")
    parser.add_argument("--config", required=True, type=Path, help="Path to the YAML config.")
    parser.add_argument("--split", default="test", help="Split alias train/val/test or a split CSV filename.")
    parser.add_argument("--hex-id", nargs="*", default=None, help="Optional hex IDs to diagnose.")
    parser.add_argument("--save-dir", default=None, help="Optional output directory for diagnostic arrays and CSV.")
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(str(args.config))
    run_patch_stitch_diagnostics(
        config=config,
        split=args.split,
        hex_ids=args.hex_id,
        save_dir=args.save_dir,
        mask_scope=args.mask_scope,
    )


if __name__ == "__main__":
    main()
