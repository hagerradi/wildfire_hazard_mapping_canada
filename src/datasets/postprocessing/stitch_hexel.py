from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StitchDiagnostics:
    coverage_count: np.ndarray
    overlap_variance: np.ndarray
    overlap_range: np.ndarray
    mean_edge_distance: np.ndarray
    summary: dict[str, float]


def _window_slices(window: np.ndarray, coord: tuple, original_shape: tuple) -> tuple[int, int, int, int, int, int]:
    r, c = int(coord[0]), int(coord[1])
    h_win, w_win = window.shape[:2]

    r_end = min(r + h_win, original_shape[0])
    c_end = min(c + w_win, original_shape[1])
    h_paste = r_end - r
    w_paste = c_end - c
    if h_paste <= 0 or w_paste <= 0:
        raise ValueError(f"Window at coord={(r, c)} does not overlap original_shape={original_shape}.")

    return r, c, r_end, c_end, h_paste, w_paste


def compute_coverage_count(coords: list[tuple], masks: list[np.ndarray], original_shape: tuple) -> np.ndarray:
    """Count how many valid window predictions contribute to each output pixel."""
    coverage = np.zeros(original_shape[:2], dtype=np.int32)

    for mask, coord in zip(masks, coords, strict=True):
        mask_arr = np.asarray(mask, dtype=bool)
        r, c, r_end, c_end, h_paste, w_paste = _window_slices(mask_arr, coord, original_shape)
        coverage[r:r_end, c:c_end] += mask_arr[:h_paste, :w_paste].astype(np.int32)

    return coverage


def _window_edge_distance(shape: tuple[int, int]) -> np.ndarray:
    rows = np.arange(shape[0])[:, np.newaxis]
    cols = np.arange(shape[1])[np.newaxis, :]
    vertical_distance = np.minimum(rows, shape[0] - 1 - rows)
    horizontal_distance = np.minimum(cols, shape[1] - 1 - cols)
    return np.minimum(vertical_distance, horizontal_distance).astype(np.float64)


def _center_crop_weight(shape: tuple[int, int], center_crop_fraction: float) -> np.ndarray:
    if not 0.0 < center_crop_fraction <= 1.0:
        raise ValueError(f"center_crop_fraction must be in (0, 1], got {center_crop_fraction}.")

    height, width = shape
    row_margin = int(np.floor((1.0 - center_crop_fraction) * height / 2.0))
    col_margin = int(np.floor((1.0 - center_crop_fraction) * width / 2.0))
    weight = np.zeros(shape, dtype=np.float64)
    weight[row_margin : height - row_margin, col_margin : width - col_margin] = 1.0
    return weight


def _feather_weight(shape: tuple[int, int], edge_epsilon: float = 1e-3) -> np.ndarray:
    if edge_epsilon <= 0.0:
        raise ValueError(f"edge_epsilon must be positive, got {edge_epsilon}.")

    height, width = shape
    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    row_center = (height - 1) / 2.0
    col_center = (width - 1) / 2.0
    row_scale = max(row_center, 1.0)
    col_scale = max(col_center, 1.0)
    row_weight = 1.0 - np.abs(rows - row_center) / row_scale
    col_weight = 1.0 - np.abs(cols - col_center) / col_scale
    weight = np.outer(np.maximum(row_weight, edge_epsilon), np.maximum(col_weight, edge_epsilon))
    return np.maximum(weight, edge_epsilon)


def _summarize_stitch_diagnostics(
    coverage: np.ndarray,
    overlap_variance: np.ndarray,
    overlap_range: np.ndarray,
    mean_edge_distance: np.ndarray,
) -> dict[str, float]:
    valid = coverage > 0
    multi = coverage > 1
    overlap_variance_values = overlap_variance[np.isfinite(overlap_variance)]
    overlap_range_values = overlap_range[np.isfinite(overlap_range)]
    edge_distance_values = mean_edge_distance[np.isfinite(mean_edge_distance)]

    return {
        "valid_pixel_count": float(np.count_nonzero(valid)),
        "zero_coverage_pixel_count": float(np.count_nonzero(coverage == 0)),
        "single_coverage_pixel_count": float(np.count_nonzero(coverage == 1)),
        "multi_coverage_pixel_count": float(np.count_nonzero(multi)),
        "max_coverage": float(np.max(coverage)) if coverage.size else 0.0,
        "mean_coverage_valid": float(np.mean(coverage[valid])) if np.any(valid) else float("nan"),
        "overlap_fraction_valid": float(np.count_nonzero(multi) / np.count_nonzero(valid)) if np.any(valid) else float("nan"),
        "mean_overlap_variance": float(np.mean(overlap_variance_values)) if overlap_variance_values.size else float("nan"),
        "p95_overlap_range": float(np.percentile(overlap_range_values, 95)) if overlap_range_values.size else float("nan"),
        "max_overlap_range": float(np.max(overlap_range_values)) if overlap_range_values.size else float("nan"),
        "mean_edge_distance_valid": float(np.mean(edge_distance_values)) if edge_distance_values.size else float("nan"),
        "p05_edge_distance_valid": float(np.percentile(edge_distance_values, 5)) if edge_distance_values.size else float("nan"),
    }


def compute_stitch_diagnostics(
    windows: list[np.ndarray],
    coords: list[tuple],
    masks: list[np.ndarray],
    original_shape: tuple,
) -> StitchDiagnostics:
    """
    Compute coverage and overlap-disagreement diagnostics for a stitched hexel.

    Overlap disagreement is measured before reducing overlapping predictions: for
    pixels covered by multiple valid windows, this reports the population variance
    and range across those contributing window values.
    """
    if len(original_shape) != 2:
        raise ValueError("Stitch diagnostics currently expect 2D output windows.")

    coverage = np.zeros(original_shape, dtype=np.int32)
    sum_arr = np.zeros(original_shape, dtype=np.float64)
    sumsq_arr = np.zeros(original_shape, dtype=np.float64)
    edge_distance_sum = np.zeros(original_shape, dtype=np.float64)
    min_arr = np.full(original_shape, np.inf, dtype=np.float64)
    max_arr = np.full(original_shape, -np.inf, dtype=np.float64)

    for window, mask, coord in zip(windows, masks, coords, strict=True):
        window_arr = np.asarray(window, dtype=np.float64)
        mask_arr = np.asarray(mask, dtype=bool)
        r, c, r_end, c_end, h_paste, w_paste = _window_slices(window_arr, coord, original_shape)

        values = window_arr[:h_paste, :w_paste]
        valid = mask_arr[:h_paste, :w_paste] & np.isfinite(values)
        edge_distance = _window_edge_distance(window_arr.shape[:2])[:h_paste, :w_paste]

        region_sum = sum_arr[r:r_end, c:c_end]
        region_sumsq = sumsq_arr[r:r_end, c:c_end]
        region_edge_distance = edge_distance_sum[r:r_end, c:c_end]
        region_count = coverage[r:r_end, c:c_end]
        region_min = min_arr[r:r_end, c:c_end]
        region_max = max_arr[r:r_end, c:c_end]

        region_sum[valid] += values[valid]
        region_sumsq[valid] += values[valid] ** 2
        region_edge_distance[valid] += edge_distance[valid]
        region_count[valid] += 1
        region_min[valid] = np.minimum(region_min[valid], values[valid])
        region_max[valid] = np.maximum(region_max[valid], values[valid])

    overlap_variance = np.full(original_shape, np.nan, dtype=np.float64)
    overlap_range = np.full(original_shape, np.nan, dtype=np.float64)
    mean_edge_distance = np.full(original_shape, np.nan, dtype=np.float64)
    multi = coverage > 1
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.divide(sum_arr, coverage, out=np.zeros_like(sum_arr), where=coverage > 0)
        variance = np.divide(sumsq_arr, coverage, out=np.zeros_like(sumsq_arr), where=coverage > 0) - mean**2
        edge_distance_mean = np.divide(
            edge_distance_sum,
            coverage,
            out=np.zeros_like(edge_distance_sum),
            where=coverage > 0,
        )

    overlap_variance[multi] = np.maximum(variance[multi], 0.0)
    overlap_range[multi] = max_arr[multi] - min_arr[multi]
    mean_edge_distance[coverage > 0] = edge_distance_mean[coverage > 0]
    summary = _summarize_stitch_diagnostics(coverage, overlap_variance, overlap_range, mean_edge_distance)

    return StitchDiagnostics(
        coverage_count=coverage,
        overlap_variance=overlap_variance,
        overlap_range=overlap_range,
        mean_edge_distance=mean_edge_distance,
        summary=summary,
    )


def stitch_windows(
    windows: list[np.ndarray],
    coords: list[tuple],
    masks: list[np.ndarray],
    original_shape: tuple,
    mode: str = "mean",
    center_crop_fraction: float = 0.8,
) -> np.ndarray:
    """
    Reconstructs an image from overlapping windows using averaging, maximization,
    center/feather weighting, or a single non-overlap contribution per pixel.

    Args:
        windows (list of np.array): List of window arrays (H_win, W_win, C).
        coords (list of tuples): List of (row, col) top-left coordinates for each window.
        masks (list of np.ndarray): List of masks for the windows. True if valid value
        original_shape (tuple): Shape of the target hexel (H, W, C).
        mode (str): How to combine/stitch the windows.
            Options: mean, max, center_crop, feathered, non_overlap.
    Returns:
        np.array: The reconstructed image (Shape: original_shape, (H,W))
    """
    dtype = np.float64

    if mode in {"mean", "center_crop", "feathered"}:
        accumulator = np.zeros(original_shape, dtype=dtype)
        counter = np.zeros(original_shape, dtype=dtype)
        fallback_accumulator = np.zeros(original_shape, dtype=dtype)
        fallback_counter = np.zeros(original_shape, dtype=dtype)

        for window, mask, coord in zip(windows, masks, coords, strict=True):
            window = np.asarray(window, dtype=dtype)
            mask = np.asarray(mask, dtype=bool)
            if mode == "center_crop":
                weights = _center_crop_weight(window.shape[:2], center_crop_fraction=center_crop_fraction)
            elif mode == "feathered":
                weights = _feather_weight(window.shape[:2])
            else:
                weights = np.ones(window.shape[:2], dtype=dtype)

            r, c, r_end, c_end, h_paste, w_paste = _window_slices(window, coord, original_shape)

            values = window[:h_paste, :w_paste]
            valid = mask[:h_paste, :w_paste] & np.isfinite(values)
            local_weights = weights[:h_paste, :w_paste] * valid
            accumulator[r:r_end, c:c_end] += np.where(valid, values * local_weights, 0.0)
            counter[r:r_end, c:c_end] += local_weights

            fallback_accumulator[r:r_end, c:c_end] += np.where(valid, values, 0.0)
            fallback_counter[r:r_end, c:c_end] += valid.astype(dtype)

        valid_mask = counter > 0
        reconstructed = np.full(original_shape, np.nan, dtype=dtype)
        reconstructed[valid_mask] = accumulator[valid_mask] / counter[valid_mask]
        fallback_only = ~valid_mask & (fallback_counter > 0)
        reconstructed[fallback_only] = fallback_accumulator[fallback_only] / fallback_counter[fallback_only]

        return reconstructed

    elif mode == "non_overlap":
        reconstructed = np.full(original_shape, np.nan, dtype=dtype)
        score = np.full(original_shape, -np.inf, dtype=dtype)

        for window, mask, coord in zip(windows, masks, coords, strict=True):
            window = np.asarray(window, dtype=dtype)
            mask = np.asarray(mask, dtype=bool)
            r, c, r_end, c_end, h_paste, w_paste = _window_slices(window, coord, original_shape)

            values = window[:h_paste, :w_paste]
            valid = mask[:h_paste, :w_paste] & np.isfinite(values)
            local_score = _window_edge_distance(window.shape[:2])[:h_paste, :w_paste]
            update = valid & (local_score > score[r:r_end, c:c_end])
            reconstructed[r:r_end, c:c_end][update] = values[update]
            score[r:r_end, c:c_end][update] = local_score[update]

        return reconstructed

    elif mode == "max":
        # --- MAX MODE ---
        # Initialize with negative infinity so any real data (even negative) will override it
        accumulator = np.full(original_shape, -np.inf, dtype=dtype)

        for window, mask, coord in zip(windows, masks, coords, strict=True):
            window = np.array(window, copy=True)
            window[~mask] = -np.inf
            r, c, r_end, c_end, h_paste, w_paste = _window_slices(window, coord, original_shape)

            # Update the area with the element-wise maximum
            current_area = accumulator[r:r_end, c:c_end]
            new_data = window[:h_paste, :w_paste]

            accumulator[r:r_end, c:c_end] = np.maximum(current_area, new_data)

        # Leave areas with no valid window contribution as NaN
        accumulator[np.isinf(accumulator)] = np.nan

        return accumulator

    else:
        raise ValueError(f"Unknown mode: {mode}")


def stitch_windows_with_diagnostics(
    windows: list[np.ndarray],
    coords: list[tuple],
    masks: list[np.ndarray],
    original_shape: tuple,
    mode: str = "mean",
    center_crop_fraction: float = 0.8,
) -> tuple[np.ndarray, StitchDiagnostics]:
    reconstructed = stitch_windows(
        windows=windows,
        coords=coords,
        masks=masks,
        original_shape=original_shape,
        mode=mode,
        center_crop_fraction=center_crop_fraction,
    )
    diagnostics = compute_stitch_diagnostics(windows=windows, coords=coords, masks=masks, original_shape=original_shape)
    return reconstructed, diagnostics
