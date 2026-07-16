"""Reusable reconstruction of patch predictions into denormalized hexel rasters.

The producer yields one target/hexel at a time so callers can stream large
buffer extents instead of materializing every reconstructed raster at once.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator
from dataclasses import dataclass

import numpy as np
import pandas as pd
from rasterio.profiles import Profile

from data_preparation.paths import Paths, normalize_mask_scope
from src.config import Config
from src.datasets.postprocessing import utils as post_utils
from src.datasets.targets import TargetSpec


@dataclass(frozen=True)
class StitchedHexel:
    """Denormalized prediction and target grids for one target on one hexel."""

    hex_id: str
    target: TargetSpec
    gt_grid: np.ndarray
    pred_grid: np.ndarray
    profile: Profile
    actual_support_mask: np.ndarray | None = None
    buffer_support_mask: np.ndarray | None = None


def load_filtered_test_metadata(config: Config, mask_scope: str) -> pd.DataFrame:
    try:
        test_df = pd.read_csv(os.path.join(config.data.root_dir, config.data.test_split))
    except (FileNotFoundError, AttributeError):
        raise ValueError("Test df file does not exist.")  # noqa: B904

    test_df = test_df[test_df["valid_ratio"] > config.data.valid_mask_threshold].reset_index(drop=True)  # type: ignore
    post_utils.validate_patch_metadata_mask_scope(test_df, mask_scope)
    return test_df


def reconstruct_denormalized_hexels(
    *,
    test_predictions: np.ndarray,
    config: Config,
    out_norm: str,
    stitch_mode: str = "mean",
    mask_scope: str = "actual",
) -> Iterator[StitchedHexel]:
    """Yield stitched, denormalized hexel grids using the same settings as training/evaluation."""
    if isinstance(test_predictions, str):
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    scope = normalize_mask_scope(mask_scope)
    settings_list = post_utils.get_target_postprocessing_settings(config=config, out_norm=out_norm)
    test_df = load_filtered_test_metadata(config=config, mask_scope=scope)
    prediction_mask_channel_indices = post_utils.get_prediction_mask_channel_indices(
        data_dir=config.data.root_dir,
        modelling_approach=config.modelling_approach,
        grid_params=post_utils.get_config_grid_params(config),
        prediction_support_policy=config.evaluation.prediction_support_policy,
    )
    patch_relative_paths = test_df["filename"].astype(str).tolist()
    grouped_hexes = list(test_df.groupby("hex_id", sort=False))
    metadata_cache_by_target: dict[int, dict[str, post_utils.PatchMetadata]] = {}
    metadata_cache_start = time.perf_counter()
    max_patch_workers = min(8, max(1, os.cpu_count() or 1))
    for settings in settings_list:
        if settings.target_channel_index in metadata_cache_by_target:
            continue
        metadata_cache_by_target[settings.target_channel_index] = post_utils.build_patch_metadata_cache(
            base_dir=config.data.root_dir,
            relative_paths=patch_relative_paths,
            target_channel_index=settings.target_channel_index,
            prediction_mask_channel_indices=prediction_mask_channel_indices,
            max_workers=max_patch_workers,
        )
    metadata_cache_elapsed = time.perf_counter() - metadata_cache_start
    print(
        "[Postprocess] Prepared patch metadata " f"for {len(set(patch_relative_paths))} patches in {metadata_cache_elapsed:.3f}s.",
        flush=True,
    )

    for raw_hex_id, one_hexel_df in grouped_hexes:
        hex_id = str(raw_hex_id).zfill(2)
        hexel_indices = one_hexel_df.index.tolist()
        hex_test_predictions = test_predictions[hexel_indices]

        for settings in settings_list:
            print(
                f"[Postprocess] Reconstructing {settings.target.name.upper()} hex {hex_id} " f"from {len(one_hexel_df)} {scope} patches...",
                flush=True,
            )
            target_predictions = post_utils.select_prediction_target_channel(
                predictions=hex_test_predictions,
                target_name=settings.target.name,
            )
            pred_grid, profile = post_utils.get_predicted_hexel(
                base_dir=config.data.root_dir,
                raw_data_dir=config.data.raw_data_dir,
                test_df=one_hexel_df,
                predictions=target_predictions,
                min_target_val=settings.min_target_val,
                max_target_val=settings.max_target_val,
                hex_id=hex_id,
                modelling_approach=config.modelling_approach,
                out_norm=settings.out_norm,
                target_log_mean=settings.target_log_mean,
                target_log_std=settings.target_log_std,
                stitch_mode=stitch_mode,
                target_channel_index=settings.target_channel_index,
                prediction_mask_channel_indices=prediction_mask_channel_indices,
                mask_scope=scope,
                patch_metadata_by_relpath=metadata_cache_by_target[settings.target_channel_index],
            )
            paths = Paths(hex_id=hex_id, root_dir=config.data.raw_data_dir)
            gt_grid, pred_grid = post_utils.load_target_grid_for_mask_scope(
                paths=paths,
                target=settings.target,
                pred_grid=pred_grid,
                profile=profile,
                mask_scope=scope,
                hex_id=hex_id,
                bp_nodata_as_zero=config.evaluation.bp_nodata_as_zero,
            )
            actual_support_mask = None
            buffer_support_mask = None
            if scope != "actual" and profile.get("crs") is not None and profile.get("transform") is not None:
                buffer_support_mask = post_utils._actual_area_mask(
                    mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=scope),
                    profile=profile,
                    shape=pred_grid.shape,
                )
                actual_support_mask = post_utils._actual_area_mask(
                    mask_path=paths.mask_grid_actual(hex_id=hex_id),
                    profile=profile,
                    shape=pred_grid.shape,
                )
            yield StitchedHexel(
                hex_id=hex_id,
                target=settings.target,
                gt_grid=gt_grid,
                pred_grid=pred_grid,
                profile=profile,
                actual_support_mask=actual_support_mask,
                buffer_support_mask=buffer_support_mask,
            )
