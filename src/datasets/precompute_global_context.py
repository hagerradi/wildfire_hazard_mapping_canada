"""Precompute downsampled full-hex input-context grids from prepared patch arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.spatial.utils import FUEL_GROUP_MAP, get_range_elevation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True, type=Path, help="Prepared data_samples directory.")
    parser.add_argument("--output-dir", required=True, type=Path, help="Directory for per-hex context .npz files.")
    parser.add_argument("--raw-data-dir", type=Path, default=None, help="Raw hexel directory for elevation normalization ranges.")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train_indices.csv", "val_indices.csv", "test_indices.csv"],
        help="Split CSVs used to find hex IDs and patch windows.",
    )
    parser.add_argument("--valid-mask-threshold", type=float, default=0.01)
    parser.add_argument("--modelling-approach", default="1")
    parser.add_argument("--context-height", type=int, default=128)
    parser.add_argument("--context-width", type=int, default=128)
    return parser.parse_args()


def load_channel_indices(data_dir: Path, modelling_approach: str) -> dict[str, int]:
    with (data_dir / f"feature_channel_map_{modelling_approach}.json").open() as f:
        feature_map = json.load(f)
    return {name: feature_map[name][0] for name in ["fuel_grid", "elevation_grid", "ignition_grid"]}


def load_split_metadata(data_dir: Path, splits: list[str], valid_mask_threshold: float) -> pd.DataFrame:
    frames = []
    for split_name in splits:
        split_path = data_dir / split_name
        split_df = pd.read_csv(split_path)
        split_df = split_df[split_df["valid_ratio"] > valid_mask_threshold].copy()
        split_df["split"] = split_name
        frames.append(split_df)
    if not frames:
        raise ValueError("No split metadata was provided.")
    metadata = pd.concat(frames, ignore_index=True)
    if metadata.empty:
        raise ValueError("No valid windows remain after valid_mask_threshold filtering.")
    return metadata


def allocate_full_hex(
    group: pd.DataFrame, data_dir: Path, indices: dict[str, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sample = np.load(data_dir / group.iloc[0]["filename"], mmap_mode="r")
    win_h, win_w = sample.shape[:2]
    full_h = int((group["row"] + win_h).max())
    full_w = int((group["col"] + win_w).max())

    sums = np.zeros((3, full_h, full_w), dtype=np.float32)
    counts = np.zeros((full_h, full_w), dtype=np.float32)

    for row in group.itertuples(index=False):
        arr = np.load(data_dir / row.filename, mmap_mode="r")
        r0 = int(row.row)
        c0 = int(row.col)
        patch = np.stack(
            [
                arr[:, :, indices["fuel_grid"]],
                arr[:, :, indices["elevation_grid"]],
                arr[:, :, indices["ignition_grid"]],
            ],
            axis=0,
        ).astype(np.float32)
        valid = np.all(np.isfinite(patch), axis=0)
        if not np.any(valid):
            continue
        r1 = r0 + patch.shape[1]
        c1 = c0 + patch.shape[2]
        sums[:, r0:r1, c0:c1] += np.where(valid[np.newaxis, :, :], patch, 0.0)
        counts[r0:r1, c0:c1] += valid.astype(np.float32)

    valid_full = counts > 0
    full = np.zeros_like(sums, dtype=np.float32)
    full[:, valid_full] = sums[:, valid_full] / counts[valid_full]
    fuel = np.rint(full[0]).astype(np.int16)
    elevation = full[1]
    ignition = full[2]
    return fuel, elevation, ignition, valid_full


def downsample_context(
    fuel: np.ndarray,
    elevation: np.ndarray,
    ignition: np.ndarray,
    valid: np.ndarray,
    context_height: int,
    context_width: int,
    elevation_min: float,
    elevation_max: float,
) -> np.ndarray:
    full_h, full_w = valid.shape
    row_bins = (np.arange(full_h) * context_height // full_h).astype(np.int64)
    col_bins = (np.arange(full_w) * context_width // full_w).astype(np.int64)
    rr, cc = np.nonzero(valid)
    br = row_bins[rr]
    bc = col_bins[cc]

    valid_count = np.zeros((context_height, context_width), dtype=np.float32)
    np.add.at(valid_count, (br, bc), 1.0)

    context_channels: list[np.ndarray] = []
    for values in [ignition, elevation]:
        value_sum = np.zeros((context_height, context_width), dtype=np.float32)
        np.add.at(value_sum, (br, bc), values[rr, cc].astype(np.float32))
        out = np.zeros_like(value_sum)
        np.divide(value_sum, valid_count, out=out, where=valid_count > 0)
        context_channels.append(out)

    context_channels[1] = np.clip((context_channels[1] - elevation_min) / (elevation_max - elevation_min + 1e-8), 0.0, 1.0)

    fuel_group_count = max(FUEL_GROUP_MAP.values()) + 1
    fuel_valid = fuel[rr, cc]
    for fuel_group in range(fuel_group_count):
        fuel_sum = np.zeros((context_height, context_width), dtype=np.float32)
        mask = fuel_valid == fuel_group
        if np.any(mask):
            np.add.at(fuel_sum, (br[mask], bc[mask]), 1.0)
        fuel_fraction = np.zeros_like(fuel_sum)
        np.divide(fuel_sum, valid_count, out=fuel_fraction, where=valid_count > 0)
        context_channels.append(fuel_fraction)

    valid_fraction = valid_count / max(float(valid_count.max()), 1.0)
    context_channels.append(valid_fraction)
    return np.stack(context_channels, axis=0).astype(np.float32)


def main() -> None:
    args = parse_args()
    indices = load_channel_indices(args.data_dir, args.modelling_approach)
    metadata = load_split_metadata(args.data_dir, args.splits, args.valid_mask_threshold)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.raw_data_dir is not None:
        elevation_max, elevation_min = get_range_elevation(str(args.raw_data_dir))
    else:
        elevation_max, elevation_min = 1.0, 0.0

    manifest_rows = []
    for hex_id, group in metadata.groupby("hex_id", sort=True):
        fuel, elevation, ignition, valid = allocate_full_hex(group, args.data_dir, indices)
        context = downsample_context(
            fuel=fuel,
            elevation=elevation,
            ignition=ignition,
            valid=valid,
            context_height=args.context_height,
            context_width=args.context_width,
            elevation_min=elevation_min,
            elevation_max=elevation_max,
        )
        hex_id_int = int(hex_id)
        out_path = args.output_dir / f"hex_{hex_id_int:02d}_global_context.npz"
        np.savez_compressed(
            out_path,
            context=context,
            full_height=np.asarray(valid.shape[0], dtype=np.int32),
            full_width=np.asarray(valid.shape[1], dtype=np.int32),
        )
        manifest_rows.append(
            {
                "hex_id": hex_id_int,
                "path": str(out_path),
                "channels": int(context.shape[0]),
                "context_height": int(context.shape[1]),
                "context_width": int(context.shape[2]),
                "full_height": int(valid.shape[0]),
                "full_width": int(valid.shape[1]),
                "valid_pixels": int(valid.sum()),
            }
        )
        print(f"Wrote {out_path}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    print(f"Wrote manifest for {len(manifest)} hexes to {args.output_dir / 'manifest.csv'}")


if __name__ == "__main__":
    main()
