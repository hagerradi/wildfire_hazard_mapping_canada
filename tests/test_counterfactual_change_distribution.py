import numpy as np

from src.datasets.postprocessing.counterfactual_change_distribution import (
    absolute_change_concentration,
    distance_bin_summary,
    summarize_change,
)


def test_change_summary_reports_concentrated_absolute_change() -> None:
    delta = np.array([[9.0, 1.0], [0.0, 0.0]], dtype=np.float32)
    paired = np.ones(delta.shape, dtype=bool)
    barrier = np.array([[True, False], [False, False]])

    summary = summarize_change(
        delta,
        paired,
        barrier,
        scenario="fuel",
        endpoint="bp",
        hex_id="16",
    )

    assert summary.original_barrier_abs_change_share == 0.9
    assert summary.top_50pct_abs_change_share == 1.0
    assert summary.pixel_share_for_80pct_abs_change == 0.25


def test_absolute_change_concentration_handles_zero_change() -> None:
    pixel_share, change_share = absolute_change_concentration(np.zeros(3))

    assert pixel_share.tolist() == [0.0, 1.0]
    assert change_share.tolist() == [0.0, 0.0]


def test_distance_summary_uses_requested_barrier_distance_bins() -> None:
    delta = np.arange(9, dtype=np.float32).reshape(3, 3)
    paired = np.ones(delta.shape, dtype=bool)
    barrier = np.zeros(delta.shape, dtype=bool)
    barrier[1, 1] = True

    summary = distance_bin_summary(
        delta,
        paired,
        barrier,
        pixel_height_m=100.0,
        pixel_width_m=100.0,
    )

    assert summary["distance_bin"].tolist() == ["100-250 m"]
    assert summary["n_pixels"].tolist() == [8]
