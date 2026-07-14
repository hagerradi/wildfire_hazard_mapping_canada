from __future__ import annotations

import numpy as np

from src.datasets.postprocessing.counterfactual_local_zoom_panels import (
    candidate_starts,
    integral_image,
    select_neighborhood_windows,
    window_sum,
)


def test_integral_image_and_window_sum_match_direct_sum() -> None:
    values = np.arange(12, dtype=np.float64).reshape(3, 4)
    integral = integral_image(values)
    assert window_sum(integral, 0, 3, 0, 4) == values.sum()
    assert window_sum(integral, 1, 3, 1, 3) == values[1:3, 1:3].sum()


def test_candidate_starts_always_includes_final_position() -> None:
    starts = candidate_starts(length=20, crop_size=6, stride=5)
    assert starts[0] == 0
    assert starts[-1] == 14
    assert all(0 <= start <= 14 for start in starts)


def test_candidate_starts_returns_zero_when_crop_exceeds_length() -> None:
    assert candidate_starts(length=5, crop_size=10, stride=2) == [0]


def test_select_neighborhood_windows_uses_positive_delta_and_barrier_density() -> None:
    barrier = np.zeros((30, 30), dtype=bool)
    barrier[2:8, 2:8] = True
    barrier[16:22, 16:22] = True
    valid = np.ones_like(barrier, dtype=bool)
    delta_hazard = np.ones((30, 30), dtype=float)
    delta_fi = np.ones((30, 30), dtype=float)
    delta_hazard[15:25, 15:25] = 5.0
    delta_fi[15:25, 15:25] = 10.0

    windows = select_neighborhood_windows(
        barrier_mask=barrier,
        valid_mask=valid,
        delta_hazard=delta_hazard,
        delta_fi=delta_fi,
        n_windows=1,
        crop_size=10,
        stride=5,
        min_barrier_pixels=10,
        min_barrier_density=0.05,
        max_barrier_density=0.8,
        exclude_border_pixels=0,
    )
    assert len(windows) == 1
    window = windows[0]
    assert window.row_min < 22 and window.row_max > 16
    assert window.col_min < 22 and window.col_max > 16
    assert window.delta_hazard_mean > 1.0


def test_select_neighborhood_windows_raises_without_candidates() -> None:
    barrier = np.zeros((10, 10), dtype=bool)
    valid = np.ones_like(barrier, dtype=bool)
    delta = np.ones((10, 10), dtype=float)
    try:
        select_neighborhood_windows(
            barrier_mask=barrier,
            valid_mask=valid,
            delta_hazard=delta,
            delta_fi=delta,
            crop_size=5,
            stride=2,
            min_barrier_pixels=1,
            exclude_border_pixels=0,
        )
    except ValueError as error:
        assert "No neighborhood windows" in str(error)
    else:
        raise AssertionError("Expected a ValueError when no barrier pixels are present.")
