"""
Generate postprocessed BP prediction rasters for use as FI/ROS scalar input channels.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import torch
import yaml

from data_preparation.paths import MASK_SCOPE_CHOICES, normalize_mask_scope
from data_preparation.spatial.utils import get_range_output
from src.config import Config, GridParams
from src.datasets.dataset import get_test_dataloader
from src.datasets.postprocessing.bp_restricted_zero import zero_nonfuel_restricted_bp
from src.datasets.postprocessing.utils import (
    get_config_grid_params,
    get_config_target_specs,
    get_mask_scope_save_dir,
    get_predicted_hexel,
    get_prediction_mask_channel_indices,
    get_target_channel_index,
    get_target_log_stats,
    get_target_out_norm,
    save_predicted_hexels,
)
from src.datasets.utils import apply_bp_nodata_zero_range, get_dataset_dimensions
from src.trainer import Trainer
from src.utils import seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BP prediction rasters for downstream BP-channel models.")
    parser.add_argument("--config", required=True, help="BP model config used for the trained checkpoint.")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to config.save_dir/evaluation.checkpoint_filename.")
    parser.add_argument("--output_dir", required=True, help="Directory where postprocessed BP prediction rasters are written.")
    parser.add_argument("--splits", nargs="+", default=["train_indices.csv", "val_indices.csv", "test_indices.csv"])
    parser.add_argument("--mask_scope", choices=MASK_SCOPE_CHOICES, default="actual")
    parser.add_argument("--stitch_mode", choices=["mean", "max", "center_crop", "feathered", "non_overlap"], default="mean")
    parser.add_argument("--center_crop_fraction", type=float, default=0.8)
    parser.add_argument("--no_restricted_postprocess", action="store_true", help="Disable non-fuel/restricted BP zeroing.")
    parser.add_argument("--no_save_patch_predictions", action="store_true", help="Do not save per-split patch prediction arrays.")
    return parser.parse_args()


def load_config(path: str) -> Config:
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _single_bp_grid_params(config: Config) -> GridParams:
    targets = get_config_target_specs(config)
    if len(targets) != 1 or targets[0].name != "bp":
        raise ValueError("BP prediction channel generation requires a single-target BP config.")
    grid_params = get_config_grid_params(config)
    if not isinstance(grid_params, GridParams):
        raise ValueError("BP prediction channel generation requires a grid source.")
    return grid_params


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(seed=getattr(config, "seed", 42), deterministic=getattr(config, "deterministic", True))
    config.logger.enabled = False

    grid_params = _single_bp_grid_params(config)
    target = get_config_target_specs(config)[0]
    checkpoint = args.checkpoint or os.path.join(config.save_dir, config.evaluation.checkpoint_filename)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_save_dir = config.save_dir
    original_test_split = config.data.test_split
    scope = normalize_mask_scope(args.mask_scope)
    scope_save_dir = get_mask_scope_save_dir(str(output_dir), scope)

    max_target_val, min_target_val = get_range_output(root_dir=config.data.raw_data_dir, output_type=target.output_type)
    max_target_val, min_target_val = apply_bp_nodata_zero_range(
        target_name=target.name,
        max_value=max_target_val,
        min_value=min_target_val,
        bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
    )
    target_log_mean, target_log_std = get_target_log_stats(grid_params=grid_params, target=target)
    target_channel_index = get_target_channel_index(config.data.root_dir, config.modelling_approach, target)
    prediction_mask_channel_indices = get_prediction_mask_channel_indices(
        data_dir=config.data.root_dir,
        modelling_approach=config.modelling_approach,
        grid_params=grid_params,
        prediction_support_policy=config.evaluation.prediction_support_policy,
    )
    out_norm = get_target_out_norm(grid_params=grid_params, target=target, fallback_out_norm=grid_params.out_norm)

    trainer: Trainer | None = None
    all_stats: list[dict] = []
    manifest_rows: list[dict] = []

    try:
        for split in args.splits:
            print(f"[BP channel] Generating predictions for split={split}, mask_scope={scope}")
            config.data.test_split = split
            test_loader = get_test_dataloader(config=config.data, modelling_approach=config.modelling_approach, seed=config.seed)
            if trainer is None:
                spatial_channels, auxiliary_input_dims = get_dataset_dimensions(test_loader.dataset)
                trainer = Trainer(config, spatial_input_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims)
                trainer.load_model(path=checkpoint, map_location=str(trainer.device))

            _, split_predictions = trainer.test(test_loader, return_predictions=True)
            if not args.no_save_patch_predictions:
                safe_split = Path(split).stem
                torch.save(
                    torch.as_tensor(split_predictions),
                    output_dir / f"{safe_split}_bp_patch_predictions.pt",
                )

            split_df = pd.read_csv(Path(config.data.root_dir) / split)
            split_df = split_df[split_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)
            for hex_raw in split_df["hex_id"].unique():
                hex_id = str(int(hex_raw)).zfill(2)
                one_hexel_df = split_df[split_df["hex_id"] == hex_raw]
                hexel_indices = split_df[split_df["hex_id"] == hex_raw].index.tolist()
                hex_predictions = split_predictions[hexel_indices]
                reconstructed, profile = get_predicted_hexel(
                    base_dir=config.data.root_dir,
                    raw_data_dir=config.data.raw_data_dir,
                    test_df=one_hexel_df,
                    predictions=hex_predictions,
                    min_target_val=min_target_val,
                    max_target_val=max_target_val,
                    hex_id=hex_id,
                    modelling_approach=config.modelling_approach,
                    out_norm=out_norm,
                    target_log_mean=target_log_mean,
                    target_log_std=target_log_std,
                    stitch_mode=args.stitch_mode,
                    target_channel_index=target_channel_index,
                    prediction_mask_channel_indices=prediction_mask_channel_indices,
                    win_h=config.data_prep.win_h,
                    win_w=config.data_prep.win_w,
                    center_crop_fraction=args.center_crop_fraction,
                    mask_scope=scope,
                )
                stats = {
                    "split": split,
                    "hex_id": hex_id,
                    "mask_scope": scope,
                    "finite_prediction_pixels": int(torch.isfinite(torch.as_tensor(reconstructed)).sum().item()),
                }
                if not args.no_restricted_postprocess:
                    reconstructed, pp_stats = zero_nonfuel_restricted_bp(
                        pred_grid=reconstructed,
                        profile=profile,
                        raw_data_dir=config.data.raw_data_dir,
                        hex_id=hex_id,
                        mask_scope=scope,
                    )
                    stats.update(pp_stats)
                    stats["split"] = split
                save_predicted_hexels(reconstructed, profile, hex_id, scope_save_dir)
                all_stats.append(stats)
                manifest_rows.append(
                    {
                        "split": split,
                        "hex_id": hex_id,
                        "prediction_path": str(Path(scope_save_dir) / "predicted_hexels" / f"hexel_{hex_id}_predicted.tif"),
                        "checkpoint": checkpoint,
                        "source_config": args.config,
                        "restricted_postprocess": not args.no_restricted_postprocess,
                    }
                )
                print(f"[BP channel] wrote hex{hex_id}")
    finally:
        config.save_dir = original_save_dir
        config.data.test_split = original_test_split

    pd.DataFrame(manifest_rows).to_csv(output_dir / "bp_prediction_channel_manifest.csv", index=False)
    pd.DataFrame(all_stats).to_csv(output_dir / "bp_prediction_channel_postprocess_stats.csv", index=False)
    print(f"[BP channel] wrote manifest and stats under {output_dir}")


if __name__ == "__main__":
    main()
