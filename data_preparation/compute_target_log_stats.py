"""
Precompute the train-only ``target_log_stats.json`` artifact.

The log1p mean/std used by the ``log_standard`` target normalization are global
constants that must be identical across training, evaluation, and inference. They are
derived solely from the training split so held-out hexes never leak into the
normalization. This mirrors the train-only spatialized-tabular imputation-stats artifact.
"""

import argparse
import logging
from pathlib import Path

from data_preparation.spatial.utils import read_split_hex_ids, write_target_log_stats

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw_data_dir",
        type=str,
        required=True,
        help="Directory holding the per-hex output rasters (e.g. canada_bp3+_2026_MILA).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Prepared patch directory containing the split index CSVs; receives the output JSON.",
    )
    parser.add_argument(
        "--train_split",
        type=str,
        default="train_indices.csv",
        help="Train split index CSV (relative to save_dir).",
    )
    parser.add_argument(
        "--output_types",
        type=str,
        nargs="+",
        default=["fire_intensity", "fire_ros"],
        help="Output types to compute log1p mean/std for.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="target_log_stats.json",
        help="Output JSON filename (relative to save_dir).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    save_dir = Path(args.save_dir)
    output_path = save_dir / args.output_file
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it.")

    train_hex_ids = read_split_hex_ids(save_dir / args.train_split)
    logger.info("Computing train-only target log-stats from %d hexes: %s", len(train_hex_ids), sorted(train_hex_ids))

    stats = write_target_log_stats(
        raw_data_dir=args.raw_data_dir,
        output_path=output_path,
        output_types=args.output_types,
        allowed_hex_ids=train_hex_ids,
    )
    logger.info("Wrote train-only target log-stats to %s: %s", output_path, stats)


if __name__ == "__main__":
    main()
