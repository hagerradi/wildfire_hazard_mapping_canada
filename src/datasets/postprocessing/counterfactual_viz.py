"""Shared IO and plotting helpers for counterfactual map scripts.

Prediction-grid raster IO (reading materialized hexel predictions, aligning to a
reference grid) and firezone boundary overlay utilities reused across the
counterfactual figure scripts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import LineCollection

DEFAULT_ZONE_OVERLAY_COLOR = "#111111"
DEFAULT_ZONE_OVERLAY_LINEWIDTH = 1.4
DEFAULT_ZONE_OVERLAY_ALPHA = 0.9


def prediction_dirs_from_index(experiment_dir: Path) -> dict[tuple[str, str], Path]:
    """Read scenario/endpoint prediction directories from the materialization index."""

    index_path = experiment_dir / "scenario_prediction_index.csv"
    index = pd.read_csv(index_path)
    required = {"scenario", "endpoint", "prediction_dir"}
    missing = sorted(required - set(index.columns))
    if missing:
        raise ValueError(f"{index_path} is missing required columns: {missing}")
    return {(str(row.scenario), str(row.endpoint)): Path(str(row.prediction_dir)) for row in index.itertuples(index=False)}


def prediction_raster_path(prediction_dir: Path, hex_id: str) -> Path:
    return prediction_dir / "predicted_hexels" / f"hexel_{int(hex_id):02d}_predicted.tif"


def read_prediction(path: Path) -> np.ma.MaskedArray:
    if not path.exists():
        raise FileNotFoundError(path)
    with rasterio.open(path) as src:
        return src.read(1, masked=True)


def prediction_reference_profile(
    prediction_dirs: dict[tuple[str, str], Path],
    hex_id: str,
    *,
    baseline_endpoint: str = "bp",
) -> dict:
    """Reference raster profile defining the prediction grid for a hexel."""

    baseline_dir = prediction_dirs.get(("baseline", baseline_endpoint))
    if baseline_dir is None:
        raise KeyError(f"Missing baseline {baseline_endpoint.upper()} prediction directory; cannot define reference grid.")
    with rasterio.open(prediction_raster_path(baseline_dir, hex_id)) as src:
        return src.profile.copy()


def add_zone_overlay_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--zone_overlay", action="store_true", help="Draw firezone boundary overlays on map panels.")
    parser.add_argument(
        "--zone_overlay_color",
        default=DEFAULT_ZONE_OVERLAY_COLOR,
        help="Firezone boundary overlay color.",
    )
    parser.add_argument(
        "--zone_overlay_linewidth",
        type=float,
        default=DEFAULT_ZONE_OVERLAY_LINEWIDTH,
        help="Firezone boundary overlay line width.",
    )
    parser.add_argument(
        "--zone_overlay_alpha",
        type=float,
        default=DEFAULT_ZONE_OVERLAY_ALPHA,
        help="Firezone boundary overlay alpha.",
    )


def zone_boundary_segments(
    zone_labels: np.ma.MaskedArray | np.ndarray,
    *,
    extent: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """Line segments tracing borders between differing (valid) firezone labels.

    Returns an ``(n, 2, 2)`` array of ``[(x0, y0), (x1, y1)]`` segments in the data
    coordinates of an ``imshow(origin="upper")`` axis: pass the same ``extent`` used
    for the base image, or ``None`` for pixel-index coordinates.  Borders touching
    masked/no-data pixels are skipped, so only firezone-firezone boundaries are drawn.
    """

    labels = np.ma.filled(np.ma.asarray(zone_labels), -1).astype(np.int64)
    valid = ~np.ma.getmaskarray(np.ma.asarray(zone_labels))
    height, width = labels.shape
    left, right, bottom, top = extent if extent is not None else (-0.5, width - 0.5, height - 0.5, -0.5)

    def to_x(edge: np.ndarray) -> np.ndarray:
        return left + edge / width * (right - left)

    def to_y(edge: np.ndarray) -> np.ndarray:
        return top + edge / height * (bottom - top)

    parts: list[np.ndarray] = []
    vertical = valid[:, :-1] & valid[:, 1:] & (labels[:, :-1] != labels[:, 1:])
    rows, cols = np.nonzero(vertical)
    if rows.size:
        x = to_x(cols + 1.0)
        start = np.column_stack([x, to_y(rows.astype(np.float64))])
        end = np.column_stack([x, to_y(rows + 1.0)])
        parts.append(np.stack([start, end], axis=1))
    horizontal = valid[:-1, :] & valid[1:, :] & (labels[:-1, :] != labels[1:, :])
    rows, cols = np.nonzero(horizontal)
    if rows.size:
        y = to_y(rows + 1.0)
        start = np.column_stack([to_x(cols.astype(np.float64)), y])
        end = np.column_stack([to_x(cols + 1.0), y])
        parts.append(np.stack([start, end], axis=1))
    if not parts:
        return np.empty((0, 2, 2), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def overlay_zone_boundaries(
    ax: plt.Axes,
    zone_labels: np.ma.MaskedArray | np.ndarray | None,
    *,
    extent: tuple[float, float, float, float] | None = None,
    color: str = DEFAULT_ZONE_OVERLAY_COLOR,
    linewidth: float = DEFAULT_ZONE_OVERLAY_LINEWIDTH,
    alpha: float = DEFAULT_ZONE_OVERLAY_ALPHA,
) -> None:
    """Overlay firezone boundaries on a hex-map axis (no-op when ``zone_labels`` is None)."""

    if zone_labels is None:
        return
    segments = zone_boundary_segments(zone_labels, extent=extent)
    if segments.shape[0] == 0:
        return
    ax.add_collection(LineCollection(list(segments), colors=color, linewidths=linewidth, alpha=alpha, zorder=5, clip_on=False))
