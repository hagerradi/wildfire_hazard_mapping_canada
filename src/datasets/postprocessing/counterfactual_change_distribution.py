"""Summarize how counterfactual prediction changes are distributed in space."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

from src.datasets.postprocessing.counterfactual_fuel_intervention_map import (
    NONFUEL_GROUP,
    group_raw_fuel,
    load_raw_fuel_on_prediction_grid,
)
from src.datasets.postprocessing.counterfactual_viz import (
    abs_share_at,
    cumulative_abs_share,
    pixel_fraction_for_share,
    prediction_dirs_from_index,
    prediction_raster_path,
    read_prediction,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SCENARIO = "remove_barriers_adjacent_modal"
DISTANCE_BIN_EDGES_M = (100.0, 250.0, 500.0, 1000.0, 2000.0, np.inf)
CONCENTRATION_FRACTIONS = (0.001, 0.01, 0.05, 0.10, 0.50)


@dataclass(frozen=True)
class ChangeSummary:
    """Per-hexel statistics on how much a scenario's prediction differs from baseline."""

    scenario: str
    endpoint: str
    hex_id: str
    n_paired_pixels: int
    delta_mean: float
    delta_median: float
    delta_abs_mean: float
    delta_sum: float
    delta_abs_sum: float
    frac_delta_positive: float
    frac_delta_negative: float
    original_barrier_abs_change_share: float
    off_barrier_abs_change_share: float
    top_0_1pct_abs_change_share: float
    top_1pct_abs_change_share: float
    top_5pct_abs_change_share: float
    top_10pct_abs_change_share: float
    top_50pct_abs_change_share: float
    pixel_share_for_50pct_abs_change: float
    pixel_share_for_80pct_abs_change: float
    pixel_share_for_90pct_abs_change: float


def _finite_prediction_values(prediction: np.ma.MaskedArray) -> np.ndarray:
    return np.asarray(prediction.filled(np.nan), dtype=np.float64)


def paired_prediction_change(
    experiment_dir: Path,
    *,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load baseline and scenario predictions and return their pixelwise delta.

    Returns:
        delta: scenario minus baseline, NaN where either prediction is missing.
        paired: mask of pixels with valid predictions in both baseline and scenario.
        profile: the baseline raster's rasterio profile (for grid/transform reuse).
    """
    prediction_dirs = prediction_dirs_from_index(experiment_dir)
    baseline_dir = prediction_dirs.get(("baseline", endpoint))
    scenario_dir = prediction_dirs.get((scenario, endpoint))
    if baseline_dir is None or scenario_dir is None:
        raise KeyError(f"Missing baseline/scenario predictions for endpoint={endpoint!r}, scenario={scenario!r}.")

    baseline_path = prediction_raster_path(baseline_dir, hex_id)
    scenario_path = prediction_raster_path(scenario_dir, hex_id)
    baseline = _finite_prediction_values(read_prediction(baseline_path))
    scenario_values = _finite_prediction_values(read_prediction(scenario_path))
    paired = np.isfinite(baseline) & np.isfinite(scenario_values)
    delta = np.full(baseline.shape, np.nan, dtype=np.float32)
    delta[paired] = (scenario_values[paired] - baseline[paired]).astype(np.float32)
    with rasterio.open(baseline_path) as src:
        profile = src.profile.copy()
    return delta, paired, profile


def original_barrier_mask(
    *,
    raw_data_dir: Path,
    reference_profile: dict,
    hex_id: str,
    support: np.ndarray,
) -> np.ndarray:
    """Return the mask of originally non-fuel ("barrier") pixels within `support`."""
    raw_fuel = load_raw_fuel_on_prediction_grid(
        raw_data_dir=raw_data_dir,
        reference_profile=reference_profile,
        hex_id=hex_id,
    )
    grouped_fuel = group_raw_fuel(raw_fuel)
    return support & np.isfinite(grouped_fuel) & (grouped_fuel == NONFUEL_GROUP)


def summarize_change(
    delta: np.ndarray,
    paired: np.ndarray,
    barrier_mask: np.ndarray,
    *,
    scenario: str,
    endpoint: str,
    hex_id: str,
) -> ChangeSummary:
    """Compute magnitude/sign/concentration statistics for a hexel's prediction change."""
    values = np.asarray(delta[paired], dtype=np.float64)
    if values.size == 0:
        raise ValueError("No paired finite prediction pixels found.")
    abs_values = np.abs(values)
    pixel_fraction, cumulative_share = cumulative_abs_share(values)
    total_abs = float(abs_values.sum())
    barrier_abs = float(np.abs(delta[barrier_mask]).sum())
    barrier_share = 0.0 if np.isclose(total_abs, 0.0) else barrier_abs / total_abs

    top_shares = {fraction: abs_share_at(pixel_fraction, cumulative_share, fraction) for fraction in CONCENTRATION_FRACTIONS}
    return ChangeSummary(
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
        n_paired_pixels=int(values.size),
        delta_mean=float(values.mean()),
        delta_median=float(np.median(values)),
        delta_abs_mean=float(abs_values.mean()),
        delta_sum=float(values.sum()),
        delta_abs_sum=total_abs,
        frac_delta_positive=float(np.mean(values > 0.0)),
        frac_delta_negative=float(np.mean(values < 0.0)),
        original_barrier_abs_change_share=float(barrier_share),
        off_barrier_abs_change_share=float(1.0 - barrier_share),
        top_0_1pct_abs_change_share=top_shares[0.001],
        top_1pct_abs_change_share=top_shares[0.01],
        top_5pct_abs_change_share=top_shares[0.05],
        top_10pct_abs_change_share=top_shares[0.10],
        top_50pct_abs_change_share=top_shares[0.50],
        pixel_share_for_50pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.50),
        pixel_share_for_80pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.80),
        pixel_share_for_90pct_abs_change=pixel_fraction_for_share(pixel_fraction, cumulative_share, 0.90),
    )


def distance_bin_summary(
    delta: np.ndarray,
    paired: np.ndarray,
    barrier_mask: np.ndarray,
    *,
    pixel_height_m: float,
    pixel_width_m: float,
) -> pd.DataFrame:
    """Bin off-barrier pixels by Euclidean distance to the nearest original barrier pixel and summarize Δ per bin."""
    if not barrier_mask.any():
        raise ValueError("No original non-fuel barrier pixels found.")
    distance_m = distance_transform_edt(
        ~barrier_mask,
        sampling=(pixel_height_m, pixel_width_m),
    )
    rows = []
    lower_bounds = DISTANCE_BIN_EDGES_M[:-1]
    upper_bounds = DISTANCE_BIN_EDGES_M[1:]
    for lower, upper in zip(lower_bounds, upper_bounds, strict=True):
        in_bin = paired & ~barrier_mask & (distance_m >= lower)
        if np.isfinite(upper):
            in_bin &= distance_m < upper
            label = f"{lower:g}-{upper:g} m"
        else:
            label = f">{lower:g} m"
        values = np.asarray(delta[in_bin], dtype=np.float64)
        if values.size == 0:
            continue
        rows.append(
            {
                "distance_bin": label,
                "distance_min_m": lower,
                "distance_max_m": upper,
                "n_pixels": int(values.size),
                "delta_mean": float(values.mean()),
                "delta_median": float(np.median(values)),
                "delta_abs_mean": float(np.abs(values).mean()),
                "delta_p25": float(np.percentile(values, 25)),
                "delta_p75": float(np.percentile(values, 75)),
                "frac_delta_positive": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


def plot_distance_summary(summary: pd.DataFrame, *, endpoint: str, scenario: str, out_path: Path) -> None:
    """Bar-plot mean/median Δ per barrier-distance bin, from `distance_bin_summary`'s output."""
    x = np.arange(len(summary))
    means = summary["delta_mean"].to_numpy(dtype=float)
    medians = summary["delta_median"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)
    bars = ax.bar(x, means, color="#4c78a8")
    ax.scatter(x, medians, color="black", marker="D", s=28, label="Median", zorder=3)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(x, summary["distance_bin"])
    ax.set_ylabel(f"Mean Δ{endpoint.upper()} (scenario − baseline)")
    ax.set_xlabel("Distance from nearest original non-fuel barrier")
    ax.set_title(f"Hex16 mean {endpoint.upper()} change by original-barrier distance")
    ax.bar_label(bars, labels=[f"n={count:,}" for count in summary["n_pixels"]], padding=4, fontsize=8)
    ax.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def write_change_distribution(
    *,
    experiment_dir: Path,
    raw_data_dir: Path,
    scenario: str,
    endpoint: str,
    hex_id: str,
    out_dir: Path | None = None,
) -> list[Path]:
    """Compute a scenario's spatial change distribution, writing the barrier-distance plot and summary CSVs.

    Writes the mean-change-by-barrier-distance plot to `out_dir` (default:
    `<experiment_dir>/figures/fuel_change_distribution`), and the change-concentration and
    barrier-distance summary CSVs to `experiment_dir`.

    Returns:
        The three paths written: [distance_plot, change_summary_csv, distance_summary_csv].
    """
    delta, paired, profile = paired_prediction_change(
        experiment_dir,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    barrier_mask = original_barrier_mask(
        raw_data_dir=raw_data_dir,
        reference_profile=profile,
        hex_id=hex_id,
        support=paired,
    )
    summary = summarize_change(
        delta,
        paired,
        barrier_mask,
        scenario=scenario,
        endpoint=endpoint,
        hex_id=hex_id,
    )
    distance_summary = distance_bin_summary(
        delta,
        paired,
        barrier_mask,
        pixel_height_m=abs(float(profile["transform"].e)),
        pixel_width_m=abs(float(profile["transform"].a)),
    )

    out_dir = out_dir or experiment_dir / "figures" / "fuel_change_distribution"
    prefix = f"hex{int(hex_id):02d}_{scenario}_{endpoint}"
    distance_path = out_dir / f"{prefix}_mean_change_by_barrier_distance.png"
    summary_path = experiment_dir / f"counterfactual_{scenario}_{endpoint}_change_concentration.csv"
    distance_summary_path = experiment_dir / f"counterfactual_{scenario}_{endpoint}_barrier_distance_summary.csv"

    plot_distance_summary(distance_summary, endpoint=endpoint, scenario=scenario, out_path=distance_path)
    pd.DataFrame([asdict(summary)]).to_csv(summary_path, index=False)
    distance_summary.to_csv(distance_summary_path, index=False)
    return [distance_path, summary_path, distance_summary_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment_dir", type=Path, default=Path("experiments/counterfactual_hex16"))
    parser.add_argument(
        "--raw_data_dir",
        type=Path,
        default=Path("/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"),
    )
    parser.add_argument("--scenario", default=SCENARIO)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--hex_id", default="16")
    parser.add_argument("--out_dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = write_change_distribution(
        experiment_dir=args.experiment_dir,
        raw_data_dir=args.raw_data_dir,
        scenario=args.scenario,
        endpoint=args.endpoint,
        hex_id=str(args.hex_id).zfill(2),
        out_dir=args.out_dir,
    )
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
