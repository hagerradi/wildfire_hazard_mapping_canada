"""Evaluate NRCAN-style fire hazard from BP and FI target/prediction rasters."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import yaml

from data_preparation.paths import MASK_SCOPE_CHOICES, Paths, normalize_mask_scope
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.postprocessing.utils import as_float_array_with_nan

HAZARD_FI_CAP = 10_000.0
HAZARD_CLASS_THRESHOLDS = np.array(
    [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0],
    dtype=np.float32,
)
DEFAULT_HIGH_CLASS_THRESHOLDS = (10, 11, 12, 13)


@dataclass
class HazardHexel:
    hex_id: str
    target_bp: np.ndarray
    target_fi: np.ndarray
    pred_bp: np.ndarray
    pred_fi: np.ndarray


@dataclass
class HazardRawHexel:
    hex_id: str
    target_raw: np.ndarray
    pred_raw: np.ndarray
    target_fi: np.ndarray
    pred_fi: np.ndarray


@dataclass(frozen=True)
class HazardEvalConfig:
    raw_data_dir: Path
    bp_pred_dir: Path
    fi_pred_dir: Path
    hex_ids: list[str]
    mask_scope: str
    save_dir: Path
    spearman_sample_size: int = 2_000_000
    seed: int = 42
    fi_caps: tuple[float | None, ...] = (HAZARD_FI_CAP,)


def parse_fi_cap_value(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"none", "null", "uncapped", "no_cap", "no-cap"}:
            return None
        cap = float(normalized)
    elif isinstance(value, int | float | np.integer | np.floating):
        cap = float(value)
    else:
        raise ValueError(f"FI cap must be positive, finite, or 'none'; got {value!r}.")
    if not np.isfinite(cap) or cap <= 0.0:
        raise ValueError(f"FI cap must be positive, finite, or 'none'; got {value!r}.")
    return cap


def parse_fi_caps(raw: dict) -> tuple[float | None, ...]:
    raw_caps = raw.get("fi_cap_sensitivity", raw.get("fi_caps", raw.get("fi_cap", HAZARD_FI_CAP)))
    if not isinstance(raw_caps, list | tuple):
        raw_caps = [raw_caps]
    caps = []
    seen = set()
    for raw_cap in raw_caps:
        cap = parse_fi_cap_value(raw_cap)
        key = "none" if cap is None else f"{cap:.12g}"
        if key in seen:
            continue
        seen.add(key)
        caps.append(cap)
    if not caps:
        raise ValueError("At least one FI cap value is required.")
    return tuple(caps)


def fi_cap_label(fi_cap: float | None) -> str:
    if fi_cap is None:
        return "uncapped"
    if float(fi_cap).is_integer():
        return f"cap{int(fi_cap)}"
    return "cap" + f"{fi_cap:g}".replace(".", "p")


def load_hazard_eval_config(path: Path) -> HazardEvalConfig:
    with path.open() as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Hazard evaluation config must be a mapping, got {type(raw).__name__}.")

    required = ("raw_data_dir", "bp_pred_dir", "fi_pred_dir", "hex_ids", "mask_scope", "save_dir")
    missing = [key for key in required if key not in raw]
    if missing:
        raise ValueError(f"Missing required hazard evaluation config keys: {missing}.")

    hex_ids = raw["hex_ids"]
    if not isinstance(hex_ids, list) or not hex_ids:
        raise ValueError("hazard evaluation config key 'hex_ids' must be a non-empty list.")

    return HazardEvalConfig(
        raw_data_dir=Path(raw["raw_data_dir"]),
        bp_pred_dir=Path(raw["bp_pred_dir"]),
        fi_pred_dir=Path(raw["fi_pred_dir"]),
        hex_ids=[str(hex_id).zfill(2) for hex_id in hex_ids],
        mask_scope=normalize_mask_scope(str(raw["mask_scope"])),
        save_dir=Path(raw["save_dir"]),
        spearman_sample_size=int(raw.get("spearman_sample_size", 2_000_000)),
        seed=int(raw.get("seed", 42)),
        fi_caps=parse_fi_caps(raw),
    )


def _as_nan_raster(path: Path) -> tuple[np.ndarray, dict]:
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype("float32")
        profile = src.profile.copy()
    return as_float_array_with_nan(arr), profile


def _prediction_path(pred_dir: Path, hex_id: str) -> Path:
    path = pred_dir / f"hexel_{hex_id}_predicted.tif"
    if not path.exists():
        raise FileNotFoundError(f"Missing predicted raster: {path}")
    return path


def compute_raw_hazard(bp: np.ndarray, fi: np.ndarray, fi_cap: float | None = HAZARD_FI_CAP) -> np.ndarray:
    bp_arr = np.clip(as_float_array_with_nan(bp), 0.0, 1.0)
    fi_arr = as_float_array_with_nan(fi)
    fi_arr = np.maximum(fi_arr, 0.0) if fi_cap is None else np.clip(fi_arr, 0.0, fi_cap)
    return (bp_arr * fi_arr).astype(np.float32)


def scale_hazard(raw_hazard: np.ndarray, hazard_max: float) -> np.ndarray:
    if not np.isfinite(hazard_max) or hazard_max <= 0.0:
        return np.full(raw_hazard.shape, np.nan, dtype=np.float32)
    return (as_float_array_with_nan(raw_hazard) * (100.0 / hazard_max)).astype(np.float32)


def classify_hazard_13(scaled_hazard: np.ndarray) -> np.ndarray:
    scaled = as_float_array_with_nan(scaled_hazard)
    classes = np.full(scaled.shape, np.nan, dtype=np.float32)
    finite = np.isfinite(scaled)
    classes[finite] = np.searchsorted(HAZARD_CLASS_THRESHOLDS, scaled[finite], side="right").astype(np.float32) + 1.0
    return classes


def _finite_pairs(pred: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred_values = np.asarray(pred, dtype=np.float64).reshape(-1)
    target_values = np.asarray(target, dtype=np.float64).reshape(-1)
    valid = np.isfinite(pred_values) & np.isfinite(target_values)
    return pred_values[valid], target_values[valid]


def _pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std == 0.0 or y_std == 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _sample_for_spearman(x: np.ndarray, y: np.ndarray, sample_size: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if sample_size <= 0 or x.size <= sample_size:
        return x, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(x.size, size=sample_size, replace=False)
    return x[idx], y[idx]


def _spearman_corr(x: np.ndarray, y: np.ndarray, sample_size: int, seed: int) -> float:
    x, y = _sample_for_spearman(x, y, sample_size=sample_size, seed=seed)
    if x.size < 2 or y.size < 2:
        return float("nan")
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return _pearson_corr(x_rank, y_rank)


def summarize_continuous(pred: np.ndarray, target: np.ndarray, prefix: str, spearman_sample_size: int, seed: int) -> dict[str, float]:
    pred_values, target_values = _finite_pairs(pred, target)
    if pred_values.size == 0:
        return {f"{prefix}_count": 0.0}
    diff = pred_values - target_values
    abs_diff = np.abs(diff)
    return {
        f"{prefix}_count": float(pred_values.size),
        f"{prefix}_target_mean": float(np.mean(target_values)),
        f"{prefix}_pred_mean": float(np.mean(pred_values)),
        f"{prefix}_bias": float(np.mean(diff)),
        f"{prefix}_mae": float(np.mean(abs_diff)),
        f"{prefix}_rmse": float(np.sqrt(np.mean(diff**2))),
        f"{prefix}_spearman": _spearman_corr(
            pred_values,
            target_values,
            sample_size=spearman_sample_size,
            seed=seed,
        ),
        f"{prefix}_target_p90": float(np.nanpercentile(target_values, 90)),
        f"{prefix}_pred_p90": float(np.nanpercentile(pred_values, 90)),
        f"{prefix}_target_p99": float(np.nanpercentile(target_values, 99)),
        f"{prefix}_pred_p99": float(np.nanpercentile(pred_values, 99)),
    }


def summarize_fi_cap_diagnostics(pred_fi: np.ndarray, target_fi: np.ndarray, fi_cap: float | None) -> dict[str, float]:
    pred_values, target_values = _finite_pairs(pred_fi, target_fi)
    metrics = {
        "fi_pair_count": float(pred_values.size),
        "target_fi_finite_count": float(np.count_nonzero(np.isfinite(target_fi))),
        "pred_fi_finite_count": float(np.count_nonzero(np.isfinite(pred_fi))),
    }
    if pred_values.size == 0:
        return metrics

    metrics.update(
        {
            "target_fi_mean": float(np.mean(target_values)),
            "pred_fi_mean": float(np.mean(pred_values)),
            "target_fi_p95": float(np.nanpercentile(target_values, 95)),
            "pred_fi_p95": float(np.nanpercentile(pred_values, 95)),
            "target_fi_p99": float(np.nanpercentile(target_values, 99)),
            "pred_fi_p99": float(np.nanpercentile(pred_values, 99)),
            "target_fi_max": float(np.nanmax(target_values)),
            "pred_fi_max": float(np.nanmax(pred_values)),
        }
    )
    if fi_cap is None:
        metrics.update(
            {
                "target_fi_gt_cap_frac": float("nan"),
                "pred_fi_gt_cap_frac": float("nan"),
                "target_fi_cap_excess_mean": float("nan"),
                "pred_fi_cap_excess_mean": float("nan"),
            }
        )
        return metrics

    target_excess = np.maximum(target_values - fi_cap, 0.0)
    pred_excess = np.maximum(pred_values - fi_cap, 0.0)
    metrics.update(
        {
            "target_fi_gt_cap_frac": float(np.mean(target_values > fi_cap)),
            "pred_fi_gt_cap_frac": float(np.mean(pred_values > fi_cap)),
            "target_fi_cap_excess_mean": float(np.mean(target_excess)),
            "pred_fi_cap_excess_mean": float(np.mean(pred_excess)),
        }
    )
    return metrics


def summarize_classes(
    pred_class: np.ndarray,
    target_class: np.ndarray,
    high_class_thresholds: tuple[int, ...] = DEFAULT_HIGH_CLASS_THRESHOLDS,
) -> dict[str, float]:
    pred_values, target_values = _finite_pairs(pred_class, target_class)
    if pred_values.size == 0:
        return {"class_count": 0.0}
    pred_int = pred_values.astype(np.int16)
    target_int = target_values.astype(np.int16)
    abs_delta = np.abs(pred_int - target_int)
    metrics = {
        "class_count": float(pred_int.size),
        "class_mae": float(np.mean(abs_delta)),
        "class_bias": float(np.mean(pred_int - target_int)),
        "class_accuracy": float(np.mean(pred_int == target_int)),
        "class_within_1_accuracy": float(np.mean(abs_delta <= 1)),
        "class_within_2_accuracy": float(np.mean(abs_delta <= 2)),
    }
    for threshold in high_class_thresholds:
        pred_high = pred_int >= threshold
        target_high = target_int >= threshold
        intersection = int(np.count_nonzero(pred_high & target_high))
        union = int(np.count_nonzero(pred_high | target_high))
        metrics[f"class_ge_{threshold}_iou"] = float(intersection / union) if union else float("nan")
        metrics[f"class_ge_{threshold}_target_frac"] = float(np.mean(target_high))
        metrics[f"class_ge_{threshold}_pred_frac"] = float(np.mean(pred_high))
    return metrics


def confusion_matrix_13(pred_class: np.ndarray, target_class: np.ndarray) -> pd.DataFrame:
    pred_values, target_values = _finite_pairs(pred_class, target_class)
    if pred_values.size == 0:
        return pd.DataFrame(columns=["target_class", "pred_class", "count"])
    frame = pd.DataFrame(
        {
            "target_class": target_values.astype(np.int16),
            "pred_class": pred_values.astype(np.int16),
        }
    )
    return frame.value_counts(["target_class", "pred_class"]).reset_index(name="count").sort_values(["target_class", "pred_class"])


def add_confusion_metadata(confusion: pd.DataFrame, group: str, scale_mode: str, fi_cap: float | None) -> pd.DataFrame:
    if confusion.empty:
        return confusion.assign(
            group=pd.Series(dtype="object"),
            scale_mode=pd.Series(dtype="object"),
            fi_cap_label=pd.Series(dtype="object"),
            fi_cap=pd.Series(dtype="float64"),
        )
    result = confusion.copy()
    result.insert(0, "fi_cap", float("nan") if fi_cap is None else fi_cap)
    result.insert(0, "fi_cap_label", fi_cap_label(fi_cap))
    result.insert(0, "scale_mode", scale_mode)
    result.insert(0, "group", group)
    return result


def load_hazard_hexel(raw_data_dir: Path, bp_pred_dir: Path, fi_pred_dir: Path, hex_id: str, mask_scope: str) -> HazardHexel:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    scope = normalize_mask_scope(mask_scope)
    mask_path = paths.mask_grid(hex_id=hex_id, mask_scope=scope)

    bp_target, bp_profile = load_spatial_raster(paths.output_burn_prob(), mask_path=mask_path)
    fi_target, _ = load_spatial_raster(paths.output_fire_intensity(), mask_path=mask_path, reference_profile=bp_profile)
    bp_pred, _ = _as_nan_raster(_prediction_path(bp_pred_dir, hex_id))
    fi_pred, _ = _as_nan_raster(_prediction_path(fi_pred_dir, hex_id))

    target_bp = as_float_array_with_nan(bp_target)
    target_fi = as_float_array_with_nan(fi_target)
    if target_bp.shape != target_fi.shape or target_bp.shape != bp_pred.shape or target_bp.shape != fi_pred.shape:
        raise ValueError(
            f"Shape mismatch for hex {hex_id}: target_bp={target_bp.shape}, target_fi={target_fi.shape}, "
            f"bp_pred={bp_pred.shape}, fi_pred={fi_pred.shape}."
        )

    return HazardHexel(
        hex_id=hex_id,
        target_bp=target_bp,
        target_fi=target_fi,
        pred_bp=bp_pred,
        pred_fi=fi_pred,
    )


def _finite_max(arrays: list[np.ndarray]) -> float:
    max_values = [float(np.nanmax(arr)) for arr in arrays if np.any(np.isfinite(arr))]
    return max(max_values) if max_values else float("nan")


def _summarize_scale_mode(
    hexels: list[HazardRawHexel],
    scale_mode: str,
    fi_cap: float | None,
    target_hazard_max: float,
    pred_hazard_max: float,
    spearman_sample_size: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if scale_mode == "product":
        target_scale_max = target_hazard_max
        pred_scale_max = pred_hazard_max
    elif scale_mode == "target_reference":
        target_scale_max = target_hazard_max
        pred_scale_max = target_hazard_max
    else:
        raise ValueError(f"Unsupported scale mode: {scale_mode!r}")

    rows = []
    all_target_scaled = []
    all_pred_scaled = []
    all_target_class = []
    all_pred_class = []
    confusion_frames = []

    for item in hexels:
        target_scaled = scale_hazard(item.target_raw, target_scale_max)
        pred_scaled = scale_hazard(item.pred_raw, pred_scale_max)
        target_class = classify_hazard_13(target_scaled)
        pred_class = classify_hazard_13(pred_scaled)

        row = {
            "group": f"hex{item.hex_id}",
            "fi_cap_label": fi_cap_label(fi_cap),
            "fi_cap": float("nan") if fi_cap is None else fi_cap,
            "scale_mode": scale_mode,
            "target_hazard_max": target_hazard_max,
            "pred_hazard_max": pred_hazard_max,
        }
        row.update(summarize_fi_cap_diagnostics(item.pred_fi, item.target_fi, fi_cap))
        row.update(summarize_continuous(item.pred_raw, item.target_raw, "raw_hazard", spearman_sample_size, seed))
        row.update(summarize_continuous(pred_scaled, target_scaled, "scaled_hazard", spearman_sample_size, seed))
        row.update(summarize_classes(pred_class, target_class))
        rows.append(row)

        confusion_frames.append(
            add_confusion_metadata(
                confusion_matrix_13(pred_class, target_class),
                group=f"hex{item.hex_id}",
                scale_mode=scale_mode,
                fi_cap=fi_cap,
            )
        )
        pred_values, target_values = _finite_pairs(pred_scaled, target_scaled)
        all_pred_scaled.append(pred_values.astype(np.float32))
        all_target_scaled.append(target_values.astype(np.float32))
        pred_cls_values, target_cls_values = _finite_pairs(pred_class, target_class)
        all_pred_class.append(pred_cls_values.astype(np.float32))
        all_target_class.append(target_cls_values.astype(np.float32))

    all_pred_scaled_arr = np.concatenate(all_pred_scaled) if all_pred_scaled else np.array([], dtype=np.float32)
    all_target_scaled_arr = np.concatenate(all_target_scaled) if all_target_scaled else np.array([], dtype=np.float32)
    all_pred_class_arr = np.concatenate(all_pred_class) if all_pred_class else np.array([], dtype=np.float32)
    all_target_class_arr = np.concatenate(all_target_class) if all_target_class else np.array([], dtype=np.float32)

    all_row = {
        "group": "all",
        "fi_cap_label": fi_cap_label(fi_cap),
        "fi_cap": float("nan") if fi_cap is None else fi_cap,
        "scale_mode": scale_mode,
        "target_hazard_max": target_hazard_max,
        "pred_hazard_max": pred_hazard_max,
    }
    all_pred_raw = []
    all_target_raw = []
    all_pred_fi = []
    all_target_fi = []
    for item in hexels:
        pred_values, target_values = _finite_pairs(item.pred_raw, item.target_raw)
        all_pred_raw.append(pred_values.astype(np.float32))
        all_target_raw.append(target_values.astype(np.float32))
        pred_fi_values, target_fi_values = _finite_pairs(item.pred_fi, item.target_fi)
        all_pred_fi.append(pred_fi_values.astype(np.float32))
        all_target_fi.append(target_fi_values.astype(np.float32))
    all_row.update(
        summarize_fi_cap_diagnostics(
            np.concatenate(all_pred_fi) if all_pred_fi else np.array([], dtype=np.float32),
            np.concatenate(all_target_fi) if all_target_fi else np.array([], dtype=np.float32),
            fi_cap,
        )
    )
    all_row.update(
        summarize_continuous(
            np.concatenate(all_pred_raw) if all_pred_raw else np.array([], dtype=np.float32),
            np.concatenate(all_target_raw) if all_target_raw else np.array([], dtype=np.float32),
            "raw_hazard",
            spearman_sample_size,
            seed,
        )
    )
    all_row.update(
        summarize_continuous(
            all_pred_scaled_arr,
            all_target_scaled_arr,
            "scaled_hazard",
            spearman_sample_size,
            seed,
        )
    )
    all_row.update(summarize_classes(all_pred_class_arr, all_target_class_arr))
    rows.append(all_row)

    confusion = confusion_matrix_13(all_pred_class_arr, all_target_class_arr)
    confusion_frames.append(add_confusion_metadata(confusion, group="all", scale_mode=scale_mode, fi_cap=fi_cap))
    class_dist = pd.concat(
        [
            pd.Series(all_target_class_arr)
            .value_counts()
            .rename_axis("class")
            .reset_index(name="target_count")
            .assign(scale_mode=scale_mode, fi_cap_label=fi_cap_label(fi_cap), fi_cap=float("nan") if fi_cap is None else fi_cap),
            pd.Series(all_pred_class_arr)
            .value_counts()
            .rename_axis("class")
            .reset_index(name="pred_count")
            .assign(scale_mode=scale_mode, fi_cap_label=fi_cap_label(fi_cap), fi_cap=float("nan") if fi_cap is None else fi_cap),
        ],
        ignore_index=True,
    )
    return pd.DataFrame(rows), pd.concat(confusion_frames, ignore_index=True), class_dist


def raw_hexels_for_cap(hexels: list[HazardHexel], fi_cap: float | None) -> list[HazardRawHexel]:
    return [
        HazardRawHexel(
            hex_id=item.hex_id,
            target_raw=compute_raw_hazard(item.target_bp, item.target_fi, fi_cap=fi_cap),
            pred_raw=compute_raw_hazard(item.pred_bp, item.pred_fi, fi_cap=fi_cap),
            target_fi=item.target_fi,
            pred_fi=item.pred_fi,
        )
        for item in hexels
    ]


def plot_confusion_matrix(confusion: pd.DataFrame, out_path: Path, title: str) -> None:
    matrix = np.zeros((13, 13), dtype=np.float64)
    if not confusion.empty:
        for row in confusion.itertuples(index=False):
            matrix[int(row.target_class) - 1, int(row.pred_class) - 1] = float(row.count)
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    im = ax.imshow(normalized, cmap="viridis", vmin=0.0, vmax=max(float(np.nanmax(normalized)), 1e-6))
    ax.set_title(title)
    ax.set_xlabel("Predicted hazard class")
    ax.set_ylabel("Target hazard class")
    ticks = np.arange(13)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(np.arange(1, 14))
    ax.set_yticklabels(np.arange(1, 14))
    fig.colorbar(im, ax=ax, label="Fraction within target class")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def evaluate_hazard(
    raw_data_dir: Path,
    bp_pred_dir: Path,
    fi_pred_dir: Path,
    hex_ids: list[str],
    mask_scope: str,
    save_dir: Path,
    spearman_sample_size: int,
    seed: int,
    fi_caps: tuple[float | None, ...] = (HAZARD_FI_CAP,),
) -> pd.DataFrame:
    hexels = [
        load_hazard_hexel(raw_data_dir=raw_data_dir, bp_pred_dir=bp_pred_dir, fi_pred_dir=fi_pred_dir, hex_id=hex_id, mask_scope=mask_scope)
        for hex_id in hex_ids
    ]
    save_dir.mkdir(parents=True, exist_ok=True)
    metrics_frames = []
    confusion_frames = []
    class_dist_frames = []
    for fi_cap in fi_caps:
        raw_hexels = raw_hexels_for_cap(hexels, fi_cap)
        target_hazard_max = _finite_max([item.target_raw for item in raw_hexels])
        pred_hazard_max = _finite_max([item.pred_raw for item in raw_hexels])
        for scale_mode in ("product", "target_reference"):
            metrics, confusion, class_dist = _summarize_scale_mode(
                hexels=raw_hexels,
                scale_mode=scale_mode,
                fi_cap=fi_cap,
                target_hazard_max=target_hazard_max,
                pred_hazard_max=pred_hazard_max,
                spearman_sample_size=spearman_sample_size,
                seed=seed,
            )
            metrics_frames.append(metrics)
            confusion_frames.append(confusion)
            class_dist_frames.append(class_dist)
            aggregate_confusion = confusion[confusion["group"] == "all"] if "group" in confusion.columns else confusion
            plot_confusion_matrix(
                confusion=aggregate_confusion,
                out_path=save_dir / f"hazard_confusion_{fi_cap_label(fi_cap)}_{scale_mode}.png",
                title=f"Hazard class confusion ({fi_cap_label(fi_cap)}, {scale_mode})",
            )
            if "group" in confusion.columns:
                for group, group_confusion in confusion.groupby("group", sort=True):
                    if group == "all":
                        continue
                    plot_confusion_matrix(
                        confusion=group_confusion,
                        out_path=save_dir / "per_hex_confusion" / f"hazard_confusion_{group}_{fi_cap_label(fi_cap)}_{scale_mode}.png",
                        title=f"Hazard class confusion ({group}, {fi_cap_label(fi_cap)}, {scale_mode})",
                    )

    metrics_df = pd.concat(metrics_frames, ignore_index=True)
    confusion_df = pd.concat(confusion_frames, ignore_index=True)
    class_dist_df = pd.concat(class_dist_frames, ignore_index=True)
    metrics_df.to_csv(save_dir / "hazard_metrics.csv", index=False)
    confusion_df.to_csv(save_dir / "hazard_confusion.csv", index=False)
    class_dist_df.to_csv(save_dir / "hazard_class_distribution.csv", index=False)
    return metrics_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="YAML postprocessing config for hazard evaluation.")
    parser.add_argument("--raw-data-dir", type=Path, default=None)
    parser.add_argument("--bp-pred-dir", type=Path, default=None)
    parser.add_argument("--fi-pred-dir", type=Path, default=None)
    parser.add_argument("--hex-ids", nargs="+", default=None)
    parser.add_argument("--mask-scope", choices=MASK_SCOPE_CHOICES, default="actual")
    parser.add_argument("--save-dir", type=Path, default=None)
    parser.add_argument("--spearman-sample-size", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fi-cap", default=str(HAZARD_FI_CAP), help="FI cap for Tom-style hazard; use 'none' for uncapped.")
    parser.add_argument(
        "--fi-cap-sensitivity",
        nargs="+",
        default=None,
        help="Optional list of FI caps to evaluate, e.g. 10000 none 5000 20000.",
    )
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> HazardEvalConfig:
    if args.config is not None:
        return load_hazard_eval_config(args.config)

    required_values = {
        "raw_data_dir": args.raw_data_dir,
        "bp_pred_dir": args.bp_pred_dir,
        "fi_pred_dir": args.fi_pred_dir,
        "hex_ids": args.hex_ids,
        "save_dir": args.save_dir,
    }
    missing = [key for key, value in required_values.items() if value is None]
    if missing:
        raise ValueError(f"Missing required CLI arguments when --config is not provided: {missing}.")

    return HazardEvalConfig(
        raw_data_dir=args.raw_data_dir,
        bp_pred_dir=args.bp_pred_dir,
        fi_pred_dir=args.fi_pred_dir,
        hex_ids=[str(hex_id).zfill(2) for hex_id in args.hex_ids],
        mask_scope=normalize_mask_scope(args.mask_scope),
        save_dir=args.save_dir,
        spearman_sample_size=args.spearman_sample_size,
        seed=args.seed,
        fi_caps=tuple(parse_fi_cap_value(value) for value in (args.fi_cap_sensitivity or [args.fi_cap])),
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)
    metrics = evaluate_hazard(
        raw_data_dir=config.raw_data_dir,
        bp_pred_dir=config.bp_pred_dir,
        fi_pred_dir=config.fi_pred_dir,
        hex_ids=config.hex_ids,
        mask_scope=config.mask_scope,
        save_dir=config.save_dir,
        spearman_sample_size=config.spearman_sample_size,
        seed=config.seed,
        fi_caps=config.fi_caps,
    )
    all_rows = metrics[metrics["group"] == "all"]
    print(all_rows.to_string(index=False))
    print(f"Wrote hazard evaluation artifacts to {config.save_dir}")


if __name__ == "__main__":
    main()
