"""Train classical pixel-level baselines for BP/FI/ROS raster targets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.ndimage import distance_transform_edt, sobel, uniform_filter
from tqdm import tqdm

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths
from data_preparation.spatial.utils import get_range_elevation, get_range_output
from src.config import Config, GridParams, TabularParams
from src.datasets.postprocessing.utils import (
    TargetPostprocessingSettings,
    calculate_hexel_metrics_pytorch,
    denormalize_model_target,
    get_config_grid_params,
    get_config_target_specs,
    get_predicted_hexel,
    get_prediction_mask_channel_indices,
    get_target_channel_index,
    get_target_log_stats,
    get_target_out_norm,
    load_target_grid_for_mask_scope,
    print_and_log_eval_metrics,
    validate_patch_metadata_mask_scope,
)
from src.datasets.targets import TargetSpec, get_target_specs
from src.datasets.utils import output_burn_prob_norm
from src.utils import AVAILABLE_METRICS, seed_everything


@dataclass(frozen=True)
class ClassicalTargetSettings:
    target: TargetSpec
    target_channel_index: int
    target_max: float
    target_min: float
    out_norm: str
    target_log_mean: float | None
    target_log_std: float | None


@dataclass(frozen=True)
class SplitArrays:
    features: np.ndarray
    targets: np.ndarray
    rows: int
    patches: int


class TabularSummaryProvider:
    """Build deterministic patch-level summaries for tabular sources."""

    _STATS = ("mean", "std", "min", "max", "p90")

    def __init__(self, config: Config, zone_channel: int):
        self.root_dir = config.data.root_dir
        self.zone_channel = zone_channel
        self.sources: list[dict[str, Any]] = []
        self.feature_names: list[str] = []

        for source in config.data.input_sources:
            if source.name == "grid" or not isinstance(source.params, TabularParams):
                continue

            params = source.params
            df = pd.read_csv(os.path.join(self.root_dir, params.csv_name))
            lut = {
                int(zone): group[params.feature_names_list].to_numpy(dtype=np.float32)
                for zone, group in df.groupby(params.fire_weather_zone_id_col)
            }
            if not lut:
                raise ValueError(f"Tabular LUT for source={source.name!r} is empty.")
            fallback = np.concatenate(list(lut.values()), axis=0)
            self.sources.append(
                {
                    "name": source.name,
                    "lut": lut,
                    "fallback": fallback,
                    "feature_names": list(params.feature_names_list),
                    "selection": params.fire_weather_zone_selection_approach,
                }
            )

            for feature_name in params.feature_names_list:
                for stat in self._STATS:
                    self.feature_names.append(f"{source.name}_{feature_name}_{stat}")
            self.feature_names.append(f"{source.name}_candidate_count")
            self.feature_names.append(f"{source.name}_used_fallback")

    def summarize(self, data: np.ndarray) -> np.ndarray:
        if not self.sources:
            return np.empty((0,), dtype=np.float32)

        zone_arr = data[:, :, self.zone_channel]
        valid_zones = zone_arr[np.isfinite(zone_arr) & (zone_arr > 0)]
        values, counts = np.unique(valid_zones.astype(np.int64), return_counts=True)

        summaries: list[float] = []
        for source in self.sources:
            candidates, used_fallback = self._select_candidates(
                values=values,
                counts=counts,
                lut=source["lut"],
                fallback=source["fallback"],
                selection=source["selection"],
            )
            summaries.extend(self._summarize_candidates(candidates))
            summaries.append(float(len(candidates)))
            summaries.append(float(used_fallback))

        return np.asarray(summaries, dtype=np.float32)

    @classmethod
    def _select_candidates(
        cls,
        values: np.ndarray,
        counts: np.ndarray,
        lut: dict[int, np.ndarray],
        fallback: np.ndarray,
        selection: str,
    ) -> tuple[np.ndarray, bool]:
        if len(values) == 0:
            return fallback, True

        if selection == "mode":
            for zone in values[np.argsort(counts)[::-1]]:
                candidates = lut.get(int(zone))
                if candidates is not None:
                    return candidates, False
            return fallback, True

        if selection == "weighted":
            all_candidates = [lut[int(zone)] for zone in values if int(zone) in lut]
            if all_candidates:
                return np.concatenate(all_candidates, axis=0), False
            return fallback, True

        raise ValueError(f"Unknown tabular zone selection approach: {selection!r}")

    @classmethod
    def _summarize_candidates(cls, candidates: np.ndarray) -> list[float]:
        stats = []
        for col in range(candidates.shape[1]):
            values = candidates[:, col]
            stats.extend(
                [
                    float(np.mean(values)),
                    float(np.std(values)),
                    float(np.min(values)),
                    float(np.max(values)),
                    float(np.percentile(values, 90)),
                ]
            )
        return stats


class ClassicalFeatureBuilder:
    """Construct per-pixel features from prepared patch arrays."""

    def __init__(self, config: Config, local_windows: Sequence[int] = (3, 7, 15)):
        self.config = config
        self.root_dir = config.data.root_dir
        self.local_windows = tuple(local_windows)
        self.grid_params = self._get_grid_params(config)
        self.targets = get_target_specs(self.grid_params.target_name)
        if len(self.targets) != 1:
            raise ValueError("The classical baseline runner currently expects a single target per config.")
        self.target = self.targets[0]

        with open(os.path.join(config.data.root_dir, f"feature_channel_map_{config.modelling_approach}.json")) as f:
            self.channel_map = json.load(f)

        self.fuel_channel = self.channel_map.get("fuel_grid", [None])[0]
        self.elevation_channel = self.channel_map.get("elevation_grid", [None])[0]
        self.ignition_channel = self.channel_map.get("ignition_grid", [None])[0]
        self.zone_channel = self.channel_map.get("firezones_grid", [None])[0]
        self.target_channel = self.channel_map[self.target.channel_key][0]

        self.elevation_max, self.elevation_min = get_range_elevation(config.data.raw_data_dir)
        target_max, target_min = get_range_output(config.data.raw_data_dir, self.target.output_type)
        target_log_mean, target_log_std = self._target_log_stats(self.target.name)
        self.target_settings = ClassicalTargetSettings(
            target=self.target,
            target_channel_index=self.target_channel,
            target_max=target_max,
            target_min=target_min,
            out_norm=self._target_out_norm(self.target.name),
            target_log_mean=target_log_mean,
            target_log_std=target_log_std,
        )

        if self.zone_channel is None:
            raise ValueError("Classical tabular summaries require 'firezones_grid' in the channel map.")
        self.tabular_provider = TabularSummaryProvider(config=config, zone_channel=int(self.zone_channel))
        self.feature_names = self._build_feature_names()

    @staticmethod
    def _get_grid_params(config: Config) -> GridParams:
        for source in config.data.input_sources:
            if source.name == "grid" and isinstance(source.params, GridParams):
                return source.params
        raise ValueError("Classical baseline requires a grid input source in the config.")

    def _target_out_norm(self, target_name: str) -> str:
        if target_name in self.grid_params.target_out_norms:
            return self.grid_params.target_out_norms[target_name]
        return self.grid_params.out_norm

    def _target_log_stats(self, target_name: str) -> tuple[float | None, float | None]:
        return (
            self.grid_params.target_log_means.get(target_name, self.grid_params.target_log_mean),
            self.grid_params.target_log_stds.get(target_name, self.grid_params.target_log_std),
        )

    def _build_feature_names(self) -> list[str]:
        names = [
            "ignition",
            "elevation_norm",
            "fuel_class",
            "fire_zone",
            "fuel_is_0",
            "fuel_is_1",
            "fuel_is_2",
            "fuel_is_3",
            "fuel_is_4",
            "fuel_is_5",
            "fuel_is_6",
            "fuel_is_7",
            "fuel_is_8",
            "fuel_is_9",
            "fuel_is_10",
            "fuel_is_11",
            "fuel_is_12",
            "fuel_is_13",
            "fuel_is_14",
            "fuel_is_15",
            "fuel_is_16",
            "fuel_is_17",
            "fuel_is_18",
            "row_norm",
            "col_norm",
            "dist_to_center",
            "patch_row",
            "patch_col",
            "global_row",
            "global_col",
            "valid_ratio",
            "zone_mode",
            "zone_mode_fraction",
            "dist_to_nonfuel",
            "ignition_gradient",
            "elevation_gradient",
        ]
        for window in self.local_windows:
            names.extend(
                [
                    f"ignition_mean_{window}",
                    f"ignition_std_{window}",
                    f"elevation_mean_{window}",
                    f"elevation_std_{window}",
                    f"elevation_tpi_{window}",
                ]
            )
        names.extend(self.tabular_provider.feature_names)
        return names

    def patch_features(self, data: np.ndarray, record: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        height, width = data.shape[:2]
        target_raw = data[:, :, self.target_channel].astype(np.float32)
        input_mask = self._input_mask(data)
        target_mask = np.isfinite(target_raw)
        mask = input_mask & target_mask

        target_filled = np.where(target_mask, target_raw, 0.0)
        target_norm = output_burn_prob_norm(
            output_arr=target_filled,
            burn_prob_max=self.target_settings.target_max,
            burn_prob_min=self.target_settings.target_min,
            out_norm=self.target_settings.out_norm,
            target_log_mean=self.target_settings.target_log_mean,
            target_log_std=self.target_settings.target_log_std,
        ).astype(np.float32)

        ignition = self._filled_channel(data, self.ignition_channel)
        elevation = self._filled_channel(data, self.elevation_channel)
        elevation_norm = (elevation - self.elevation_min) / (self.elevation_max - self.elevation_min + 1e-8)
        fuel = np.rint(self._filled_channel(data, self.fuel_channel, fill_value=0.0)).astype(np.int16)
        fuel = np.clip(fuel, 0, 18)
        fire_zone = self._filled_channel(data, self.zone_channel, fill_value=0.0)

        rows = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None] * np.ones((height, width), dtype=np.float32)
        cols = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :] * np.ones((height, width), dtype=np.float32)
        patch_row = float(record.get("row", 0.0))
        patch_col = float(record.get("col", 0.0))
        valid_ratio = float(record.get("valid_ratio", np.mean(mask)))
        zone_mode, zone_fraction = self._zone_mode_and_fraction(data)

        feature_arrays: list[np.ndarray] = [
            ignition,
            elevation_norm,
            fuel.astype(np.float32),
            fire_zone,
        ]
        feature_arrays.extend((fuel == cls).astype(np.float32) for cls in range(19))
        feature_arrays.extend(
            [
                rows,
                cols,
                np.sqrt((rows - 0.5) ** 2 + (cols - 0.5) ** 2).astype(np.float32),
                np.full((height, width), patch_row, dtype=np.float32),
                np.full((height, width), patch_col, dtype=np.float32),
                (patch_row + np.arange(height, dtype=np.float32)[:, None]) * np.ones((height, width), dtype=np.float32),
                (patch_col + np.arange(width, dtype=np.float32)[None, :]) * np.ones((height, width), dtype=np.float32),
                np.full((height, width), valid_ratio, dtype=np.float32),
                np.full((height, width), zone_mode, dtype=np.float32),
                np.full((height, width), zone_fraction, dtype=np.float32),
                self._distance_to_nonfuel(fuel),
                self._gradient_magnitude(ignition),
                self._gradient_magnitude(elevation_norm),
            ]
        )

        for window in self.local_windows:
            ignition_mean = self._nanmean_filter(ignition, window)
            elevation_mean = self._nanmean_filter(elevation_norm, window)
            feature_arrays.extend(
                [
                    ignition_mean,
                    self._nanstd_filter(ignition, window, ignition_mean),
                    elevation_mean,
                    self._nanstd_filter(elevation_norm, window, elevation_mean),
                    elevation_norm - elevation_mean,
                ]
            )

        tabular_summary = self.tabular_provider.summarize(data)
        feature_arrays.extend(np.full((height, width), value, dtype=np.float32) for value in tabular_summary)

        features = np.stack(feature_arrays, axis=-1).astype(np.float32)
        if features.shape[-1] != len(self.feature_names):
            raise RuntimeError(f"Feature name mismatch: got {features.shape[-1]} arrays for {len(self.feature_names)} names.")

        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        return features, target_norm, mask

    def _input_mask(self, data: np.ndarray) -> np.ndarray:
        channels = []
        for feature_name in self.grid_params.feature_names_list:
            channel_indices = self.channel_map.get(feature_name)
            if not channel_indices:
                raise ValueError(f"Missing configured grid feature {feature_name!r} in channel map.")
            channels.extend(channel_indices)
        return np.all(np.isfinite(data[:, :, channels]), axis=-1)

    @staticmethod
    def _filled_channel(data: np.ndarray, channel: int | None, fill_value: float | None = None) -> np.ndarray:
        if channel is None:
            raise ValueError("Required feature channel is missing from the channel map.")
        arr = data[:, :, channel].astype(np.float32)
        if fill_value is None:
            finite = np.isfinite(arr)
            fill_value = float(np.nanmean(arr)) if np.any(finite) else 0.0
        return np.where(np.isfinite(arr), arr, fill_value).astype(np.float32)

    @staticmethod
    def _nanmean_filter(arr: np.ndarray, window: int) -> np.ndarray:
        finite = np.isfinite(arr)
        filled = np.where(finite, arr, 0.0).astype(np.float32)
        numerator = uniform_filter(filled, size=window, mode="nearest")
        denominator = uniform_filter(finite.astype(np.float32), size=window, mode="nearest")
        return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-8).astype(np.float32)

    @classmethod
    def _nanstd_filter(cls, arr: np.ndarray, window: int, mean: np.ndarray | None = None) -> np.ndarray:
        mean = cls._nanmean_filter(arr, window) if mean is None else mean
        mean_sq = cls._nanmean_filter(arr * arr, window)
        return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)

    @staticmethod
    def _gradient_magnitude(arr: np.ndarray) -> np.ndarray:
        grad_y = sobel(arr, axis=0, mode="nearest")
        grad_x = sobel(arr, axis=1, mode="nearest")
        return np.hypot(grad_y, grad_x).astype(np.float32)

    @staticmethod
    def _distance_to_nonfuel(fuel: np.ndarray) -> np.ndarray:
        nonfuel = fuel == 0
        if not np.any(nonfuel):
            return np.ones_like(fuel, dtype=np.float32)
        distance = distance_transform_edt(~nonfuel)
        return (distance / max(fuel.shape)).astype(np.float32)

    def _zone_mode_and_fraction(self, data: np.ndarray) -> tuple[float, float]:
        zone_arr = data[:, :, int(self.zone_channel)]
        valid = zone_arr[np.isfinite(zone_arr) & (zone_arr > 0)]
        if valid.size == 0:
            return 0.0, 0.0
        values, counts = np.unique(valid.astype(np.int64), return_counts=True)
        max_idx = int(np.argmax(counts))
        return float(values[max_idx]), float(counts[max_idx] / valid.size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a classical pixel-level LightGBM baseline.")
    parser.add_argument("--config", required=True, help="Path to an existing BP/FI/ROS YAML config.")
    parser.add_argument("--save-dir", default=None, help="Override config.save_dir for classical artifacts.")
    parser.add_argument("--objective", default="auto", choices=["auto", "regression", "huber", "cross_entropy", "quantile"])
    parser.add_argument("--quantile-alpha", type=float, default=0.95)
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=127)
    parser.add_argument("--min-child-samples", type=int, default=100)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--reg-alpha", type=float, default=0.1)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--num-threads", type=int, default=0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--train-pixels-per-patch", type=int, default=512)
    parser.add_argument("--val-pixels-per-patch", type=int, default=512)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-val-rows", type=int, default=0)
    parser.add_argument("--max-train-patches", type=int, default=0)
    parser.add_argument("--max-eval-patches", type=int, default=0)
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    parser.add_argument("--tail-fraction", type=float, default=0.5)
    parser.add_argument("--tail-quantile", type=float, default=0.9)
    parser.add_argument("--local-windows", default="3,7,15")
    parser.add_argument("--batch-predict-rows", type=int, default=200_000)
    parser.add_argument("--skip-hexel-eval", action="store_true", help="Only compute patch metrics.")
    parser.add_argument("--save-test-predictions", action="store_true", help="Persist test_predictions.npy under save_dir.")
    return parser.parse_args()


def load_config(path: str) -> Config:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        raw = yaml.safe_load(f)
    return Config(**raw)


def _read_split(config: Config, split_name: str) -> pd.DataFrame:
    split_path = os.path.join(config.data.root_dir, split_name)
    df = pd.read_csv(split_path)
    if "valid_ratio" in df.columns:
        df = df[df["valid_ratio"] > config.data.valid_mask_threshold].copy()
    return df.reset_index(drop=True)


def _sample_patch_rows(
    features: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    rows_per_patch: int,
    rng: np.random.Generator,
    tail_fraction: float,
    tail_quantile: float,
) -> tuple[np.ndarray, np.ndarray]:
    valid_indices = np.flatnonzero(mask.reshape(-1))
    if valid_indices.size == 0:
        return np.empty((0, features.shape[-1]), dtype=np.float32), np.empty((0,), dtype=np.float32)

    if rows_per_patch <= 0 or rows_per_patch >= valid_indices.size:
        chosen = valid_indices
    else:
        target_flat = target.reshape(-1)
        tail_count = int(round(rows_per_patch * tail_fraction))
        tail_count = max(0, min(tail_count, rows_per_patch))
        random_count = rows_per_patch - tail_count

        chosen_parts = []
        if tail_count > 0:
            threshold = np.quantile(target_flat[valid_indices], tail_quantile)
            tail_pool = valid_indices[target_flat[valid_indices] >= threshold]
            if tail_pool.size > 0:
                chosen_parts.append(rng.choice(tail_pool, size=min(tail_count, tail_pool.size), replace=False))

        if random_count > 0:
            chosen_parts.append(rng.choice(valid_indices, size=min(random_count, valid_indices.size), replace=False))

        chosen = np.unique(np.concatenate(chosen_parts)) if chosen_parts else np.empty((0,), dtype=np.int64)
        if chosen.size < min(rows_per_patch, valid_indices.size):
            remaining = np.setdiff1d(valid_indices, chosen, assume_unique=False)
            fill_count = min(rows_per_patch, valid_indices.size) - chosen.size
            if remaining.size > 0 and fill_count > 0:
                chosen = np.concatenate([chosen, rng.choice(remaining, size=min(fill_count, remaining.size), replace=False)])

    features_2d = features.reshape(-1, features.shape[-1])
    target_1d = target.reshape(-1)
    return features_2d[chosen].astype(np.float32), target_1d[chosen].astype(np.float32)


def extract_split_rows(
    config: Config,
    builder: ClassicalFeatureBuilder,
    split_name: str,
    rows_per_patch: int,
    rng: np.random.Generator,
    tail_fraction: float,
    tail_quantile: float,
    max_rows: int = 0,
    max_patches: int = 0,
) -> SplitArrays:
    df = _read_split(config, split_name)
    if max_patches > 0:
        df = df.iloc[:max_patches].copy()

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for record in tqdm(df.to_dict("records"), desc=f"Extracting {split_name}", leave=True):
        data = np.load(os.path.join(config.data.root_dir, record[config.data.filename_col]), mmap_mode="r")
        features, target, mask = builder.patch_features(data=data, record=record)
        x_patch, y_patch = _sample_patch_rows(
            features=features,
            target=target,
            mask=mask,
            rows_per_patch=rows_per_patch,
            rng=rng,
            tail_fraction=tail_fraction,
            tail_quantile=tail_quantile,
        )
        if len(y_patch) == 0:
            continue
        x_parts.append(x_patch)
        y_parts.append(y_patch)

        if max_rows > 0 and sum(len(part) for part in y_parts) >= max_rows:
            break

    if not x_parts:
        raise ValueError(f"No training rows extracted from split {split_name!r}.")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if max_rows > 0 and len(y) > max_rows:
        selected = rng.choice(np.arange(len(y)), size=max_rows, replace=False)
        x = x[selected]
        y = y[selected]

    return SplitArrays(features=x, targets=y, rows=len(y), patches=len(x_parts))


def _auto_objective(target: TargetSpec, out_norm: str) -> str:
    if target.probability_scale and out_norm == "min_max":
        return "cross_entropy"
    if out_norm == "log_standard":
        return "huber"
    return "regression"


def train_lightgbm(
    args: argparse.Namespace,
    builder: ClassicalFeatureBuilder,
    train_arrays: SplitArrays,
    val_arrays: SplitArrays,
) -> lgb.LGBMRegressor:
    objective = _auto_objective(builder.target, builder.target_settings.out_norm) if args.objective == "auto" else args.objective
    params: dict[str, Any] = {
        "objective": objective,
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_child_samples": args.min_child_samples,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "reg_alpha": args.reg_alpha,
        "reg_lambda": args.reg_lambda,
        "n_jobs": args.num_threads if args.num_threads != 0 else -1,
        "random_state": builder.config.seed,
        "verbosity": -1,
    }
    if objective == "quantile":
        params["alpha"] = args.quantile_alpha

    model = lgb.LGBMRegressor(**params)
    callbacks: list[Callable] = [lgb.log_evaluation(period=50)]
    if args.early_stopping_rounds > 0 and val_arrays.rows > 0:
        callbacks.append(lgb.early_stopping(stopping_rounds=args.early_stopping_rounds))

    model.fit(
        train_arrays.features,
        train_arrays.targets,
        eval_set=[(val_arrays.features, val_arrays.targets)],
        eval_names=["val"],
        callbacks=callbacks,
        feature_name=builder.feature_names,
    )
    return model


def predict_split(
    config: Config,
    builder: ClassicalFeatureBuilder,
    model: lgb.LGBMRegressor,
    split_name: str,
    batch_rows: int,
    max_patches: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    df = _read_split(config, split_name)
    if max_patches > 0:
        df = df.iloc[:max_patches].copy()

    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    masks: list[np.ndarray] = []

    for record in tqdm(df.to_dict("records"), desc=f"Predicting {split_name}", leave=True):
        data = np.load(os.path.join(config.data.root_dir, record[config.data.filename_col]), mmap_mode="r")
        features, target, mask = builder.patch_features(data=data, record=record)
        pred = np.zeros(target.shape, dtype=np.float32)
        valid_indices = np.flatnonzero(mask.reshape(-1))
        if valid_indices.size > 0:
            features_2d = features.reshape(-1, features.shape[-1])
            valid_features = features_2d[valid_indices]
            pred_valid = np.empty((valid_indices.size,), dtype=np.float32)
            booster = model.booster_
            for start in range(0, valid_indices.size, batch_rows):
                end = min(start + batch_rows, valid_indices.size)
                batch_pred = booster.predict(valid_features[start:end], num_iteration=model.best_iteration_)
                pred_valid[start:end] = np.asarray(batch_pred, dtype=np.float32)
            pred.reshape(-1)[valid_indices] = _clip_model_domain_predictions(pred_valid, builder.target_settings.out_norm)

        predictions.append(pred)
        targets.append(target.astype(np.float32))
        masks.append(mask.astype(bool))

    return np.stack(predictions, axis=0), np.stack(targets, axis=0), np.stack(masks, axis=0)


def _clip_model_domain_predictions(predictions: np.ndarray, out_norm: str) -> np.ndarray:
    if out_norm in {"min_max", "log"}:
        return np.clip(predictions, 0.0, 1.0)
    return predictions


def compute_patch_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    masks: np.ndarray,
    settings: ClassicalTargetSettings,
    metric_functions: dict[str, Callable],
    batch_size: int = 32,
) -> dict[str, float]:
    running = {name: 0.0 for name in metric_functions}
    count = 0
    for start in range(0, len(predictions), batch_size):
        end = min(start + batch_size, len(predictions))
        pred_phys = denormalize_model_target(
            data=predictions[start:end],
            min_val=settings.target_min,
            max_val=settings.target_max,
            out_norm=settings.out_norm,
            target_log_mean=settings.target_log_mean,
            target_log_std=settings.target_log_std,
        )
        target_phys = denormalize_model_target(
            data=targets[start:end],
            min_val=settings.target_min,
            max_val=settings.target_max,
            out_norm=settings.out_norm,
            target_log_mean=settings.target_log_mean,
            target_log_std=settings.target_log_std,
        )
        pred_tensor = torch.from_numpy(pred_phys).float().unsqueeze(1)
        target_tensor = torch.from_numpy(target_phys).float().unsqueeze(1)
        mask_tensor = torch.from_numpy(masks[start:end]).bool().unsqueeze(1)
        current_batch = end - start
        count += current_batch
        with torch.no_grad():
            for name, metric_fn in metric_functions.items():
                value = metric_fn(pred_tensor, target_tensor, mask_tensor)
                running[name] += float(value.item()) * current_batch

    return {name: value / max(count, 1) for name, value in running.items()}


def evaluate_hexel_metrics_only(
    predictions: np.ndarray,
    config: Config,
    metric_functions: dict[str, Callable],
    mask_scope: str = "actual",
) -> dict[str, float]:
    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    modelling_approach = config.modelling_approach
    targets = get_config_target_specs(config)
    grid_params = get_config_grid_params(config)
    if len(targets) != 1:
        raise ValueError("Classical hexel evaluation currently expects a single target.")
    target = targets[0]

    target_channel_index = get_target_channel_index(data_dir=data_dir, modelling_approach=modelling_approach, target=target)
    prediction_mask_channel_indices = get_prediction_mask_channel_indices(
        data_dir=data_dir,
        modelling_approach=modelling_approach,
        grid_params=grid_params,
    )
    max_target_val, min_target_val = get_range_output(root_dir=raw_data_dir, output_type=target.output_type)
    target_log_mean, target_log_std = get_target_log_stats(grid_params=grid_params, target=target)
    settings = TargetPostprocessingSettings(
        target=target,
        target_channel_index=target_channel_index,
        max_target_val=max_target_val,
        min_target_val=min_target_val,
        out_norm=get_target_out_norm(grid_params=grid_params, target=target, fallback_out_norm="min_max"),
        target_log_mean=target_log_mean,
        target_log_std=target_log_std,
    )

    test_df = _read_split(config, config.data.test_split)
    validate_patch_metadata_mask_scope(test_df, mask_scope)
    all_hexel_metrics: list[tuple[str, dict[str, float]]] = []

    for hex_id_value in test_df["hex_id"].unique():
        hex_id = str(hex_id_value).zfill(2)
        one_hexel_df = test_df[test_df["hex_id"] == hex_id_value]
        hexel_indices = one_hexel_df.index.tolist()
        reconstructed_hexel_denorm, gt_profile = get_predicted_hexel(
            base_dir=data_dir,
            raw_data_dir=raw_data_dir,
            test_df=one_hexel_df,
            predictions=predictions[hexel_indices],
            min_target_val=settings.min_target_val,
            max_target_val=settings.max_target_val,
            hex_id=hex_id,
            modelling_approach=modelling_approach,
            out_norm=settings.out_norm,
            target_log_mean=settings.target_log_mean,
            target_log_std=settings.target_log_std,
            stitch_mode="mean",
            target_channel_index=settings.target_channel_index,
            prediction_mask_channel_indices=prediction_mask_channel_indices,
            win_h=config.data_prep.win_h,
            win_w=config.data_prep.win_w,
            mask_scope=mask_scope,
        )
        all_paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
        gt_grid, reconstructed_hexel_denorm = load_target_grid_for_mask_scope(
            paths=all_paths,
            target=target,
            pred_grid=reconstructed_hexel_denorm,
            profile=gt_profile,
            mask_scope=mask_scope,
            hex_id=hex_id,
        )
        hex_metrics = calculate_hexel_metrics_pytorch(
            gt_grid=gt_grid,
            pred_grid=reconstructed_hexel_denorm,
            device=torch.device("cpu"),
            metric_functions=metric_functions,
        )
        all_hexel_metrics.append((hex_id, hex_metrics))

    hexel_metrics: dict[str, float] = {}
    metric_values_by_key: dict[str, list[float]] = {}
    for hex_id, metrics in all_hexel_metrics:
        for key, value in metrics.items():
            metric_value = float(value) if not np.isnan(value) else float("nan")
            hexel_metrics[f"hex{hex_id}/{key}"] = metric_value
            metric_values_by_key.setdefault(key, []).append(metric_value)

    for key, values in metric_values_by_key.items():
        finite_values = [value for value in values if not np.isnan(value)]
        hexel_metrics[f"all/{key}"] = float(np.mean(finite_values)) if finite_values else float("nan")

    return hexel_metrics


def save_artifacts(
    save_dir: Path,
    model: lgb.LGBMRegressor,
    builder: ClassicalFeatureBuilder,
    args: argparse.Namespace,
    train_arrays: SplitArrays,
    val_arrays: SplitArrays,
    test_metrics: dict[str, float],
    hexel_metrics: dict[str, float],
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(save_dir / "model.txt"))
    with (save_dir / "feature_names.json").open("w") as f:
        json.dump(builder.feature_names, f, indent=2)
    with (save_dir / "classical_run_summary.json").open("w") as f:
        json.dump(
            {
                "args": vars(args),
                "target": builder.target.name,
                "out_norm": builder.target_settings.out_norm,
                "train_rows": train_arrays.rows,
                "train_patches": train_arrays.patches,
                "val_rows": val_arrays.rows,
                "val_patches": val_arrays.patches,
                "test_metrics": test_metrics,
                "hexel_metrics": hexel_metrics,
            },
            f,
            indent=2,
        )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.save_dir is not None:
        config.save_dir = args.save_dir
    else:
        config.save_dir = os.path.join(config.save_dir, "classical_lightgbm")

    seed_everything(config.seed, deterministic=config.deterministic)
    rng = np.random.default_rng(config.seed)
    local_windows = tuple(int(item) for item in args.local_windows.split(",") if item)
    metric_functions = {name: AVAILABLE_METRICS[name] for name in config.metrics}

    builder = ClassicalFeatureBuilder(config=config, local_windows=local_windows)
    print(f"[Classical] Target={builder.target.name} out_norm={builder.target_settings.out_norm}")
    print(f"[Classical] Feature count={len(builder.feature_names)}")

    train_arrays = extract_split_rows(
        config=config,
        builder=builder,
        split_name=config.data.train_split,
        rows_per_patch=args.train_pixels_per_patch,
        rng=rng,
        tail_fraction=args.tail_fraction,
        tail_quantile=args.tail_quantile,
        max_rows=args.max_train_rows,
        max_patches=args.max_train_patches,
    )
    val_arrays = extract_split_rows(
        config=config,
        builder=builder,
        split_name=config.data.val_split,
        rows_per_patch=args.val_pixels_per_patch,
        rng=rng,
        tail_fraction=args.tail_fraction,
        tail_quantile=args.tail_quantile,
        max_rows=args.max_val_rows,
        max_patches=args.max_eval_patches,
    )
    print(f"[Classical] Train rows={train_arrays.rows:,} from patches={train_arrays.patches:,}")
    print(f"[Classical] Val rows={val_arrays.rows:,} from patches={val_arrays.patches:,}")

    model = train_lightgbm(args=args, builder=builder, train_arrays=train_arrays, val_arrays=val_arrays)

    test_predictions, test_targets, test_masks = predict_split(
        config=config,
        builder=builder,
        model=model,
        split_name=config.data.test_split,
        batch_rows=args.batch_predict_rows,
        max_patches=args.max_eval_patches,
    )
    test_metrics = compute_patch_metrics(
        predictions=test_predictions,
        targets=test_targets,
        masks=test_masks,
        settings=builder.target_settings,
        metric_functions=metric_functions,
    )
    hexel_metrics = (
        {}
        if args.skip_hexel_eval or args.max_eval_patches > 0
        else evaluate_hexel_metrics_only(test_predictions, config, metric_functions, mask_scope=args.mask_scope)
    )

    save_dir = Path(config.save_dir)
    save_artifacts(
        save_dir=save_dir,
        model=model,
        builder=builder,
        args=args,
        train_arrays=train_arrays,
        val_arrays=val_arrays,
        test_metrics=test_metrics,
        hexel_metrics=hexel_metrics,
    )
    if args.save_test_predictions:
        np.save(save_dir / "test_predictions.npy", test_predictions.astype(np.float32))

    print_and_log_eval_metrics(test_metrics=test_metrics, hexel_metrics=hexel_metrics, experiment_logger=None)
    print(f"\n[Classical] Saved artifacts to {save_dir}")


if __name__ == "__main__":
    main()
