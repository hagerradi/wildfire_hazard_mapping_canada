"""Prepare patch metadata for an explicit list of hexes."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from data_preparation.hexel_loader import load_spatial_features_per_hexel
from data_preparation.paths import MASK_SCOPE_CHOICES
from data_preparation.process_hexels_into_grids import get_split_hexel_window


def normalize_hex_id(hex_id: str) -> str:
    return hex_id.removeprefix("hex").zfill(2)


def copy_shared_tables(source_data_dir: Path, out_dir: Path) -> None:
    for filename in ("weather_table_processed.csv", "df_fire_fru_processed.csv"):
        src = source_data_dir / filename
        dst = out_dir / filename
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)


def prepare_selected_hexel_patches(
    raw_root: Path,
    out_dir: Path,
    hex_ids: list[str],
    source_data_dir: Path | None = None,
    split_name: str = "test_indices.csv",
    mask_scope: str = "actual",
    modelling_approach: int = 1,
    win_h: int = 128,
    win_w: int = 128,
    overlap_ratio: float = 0.2,
) -> pd.DataFrame:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "numpy_files").mkdir(exist_ok=True)
    if source_data_dir is not None:
        copy_shared_tables(source_data_dir=source_data_dir, out_dir=out_dir)

    feature_map = out_dir / f"feature_channel_map_{modelling_approach}.json"
    frames = []
    for hex_id in sorted({normalize_hex_id(hex_id) for hex_id in hex_ids}):
        meta_path = out_dir / f"meta_hex_{hex_id}.csv"
        if not meta_path.exists():
            stacked, mask, season_cause_mapping = load_spatial_features_per_hexel(
                root_dir=str(raw_root),
                hex_id=hex_id,
                feature_channel_map_path=str(feature_map),
                modelling_approach=modelling_approach,
                mask_scope=mask_scope,
            )
            if stacked is None or mask is None:
                raise RuntimeError(f"Failed to load features for hex{hex_id}.")
            get_split_hexel_window(
                season_cause_stacked_feats=stacked,
                season_cause_mask=mask,
                season_cause_mapping=season_cause_mapping,
                out_dir=str(out_dir),
                root_dir=str(raw_root),
                hex_id=hex_id,
                win_h=win_h,
                win_w=win_w,
                overlap_ratio=overlap_ratio,
                mask_scope=mask_scope,
            )
        frame = pd.read_csv(meta_path)
        frame["hex_id"] = frame["hex_id"].astype(str).str.zfill(2)
        frames.append(frame)

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(out_dir / split_name, index=False)
    return combined


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--hex-ids", nargs="+", required=True)
    parser.add_argument("--source-data-dir", type=Path, default=None)
    parser.add_argument("--split-name", default="test_indices.csv")
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    parser.add_argument("--modelling-approach", type=int, default=1)
    parser.add_argument("--win-h", type=int, default=128)
    parser.add_argument("--win-w", type=int, default=128)
    parser.add_argument("--overlap-ratio", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    combined = prepare_selected_hexel_patches(
        raw_root=args.raw_root,
        out_dir=args.out_dir,
        hex_ids=args.hex_ids,
        source_data_dir=args.source_data_dir,
        split_name=args.split_name,
        mask_scope=args.mask_scope,
        modelling_approach=args.modelling_approach,
        win_h=args.win_h,
        win_w=args.win_w,
        overlap_ratio=args.overlap_ratio,
    )
    print(f"Wrote {args.out_dir / args.split_name} rows={len(combined)}")
    print(combined.groupby("hex_id")["filename"].count().to_string())


if __name__ == "__main__":
    main()
