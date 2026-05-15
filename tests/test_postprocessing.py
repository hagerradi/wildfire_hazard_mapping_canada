import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.utils import denormalize_model_target, get_hexel_binary_maps, get_stitched_windows
from src.datasets.postprocessing.visualize_predictions import get_distribution_axis_limit
from src.datasets.utils import output_target_norm


def test_get_stitched_windows_uses_target_channel_mask(tmp_path):
    patch = np.ones((2, 2, 7), dtype=np.float32)
    patch[:, :, 0] = 1.0
    patch[:, :, 5] = 1.0
    patch[0, 1, 5] = np.nan
    np.save(tmp_path / "sample.npy", patch)

    df = pd.DataFrame([["sample.npy", None, None, None, None, 0, 0]])
    predictions = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stitched = get_stitched_windows(
        base_dir=str(tmp_path),
        df=df,
        predictions=predictions,
        start_idx=0,
        gt_shape=(2, 2),
        target_channel_index=5,
        win_h=2,
        win_w=2,
    )

    assert stitched[0, 0] == 1.0
    assert np.isnan(stitched[0, 1])
    assert stitched[1, 0] == 3.0
    assert stitched[1, 1] == 4.0


def test_distribution_axis_limit_uses_non_probability_fallback():
    empty = np.array([], dtype=np.float32)

    assert get_distribution_axis_limit(empty, empty, probability_scale=True) == 0.15
    assert get_distribution_axis_limit(empty, empty, probability_scale=False) == 1.0


def test_distribution_axis_limit_only_caps_probability_scale():
    gt_vals = np.array([0.0, 2.0], dtype=np.float32)
    pred_vals = np.array([3.0], dtype=np.float32)

    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=True) == 1.0
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) > 3.0


def test_log_standard_target_transform_roundtrip():
    raw = np.array([[0.0, 1.0, 9.0]], dtype=np.float32)
    mean = 1.25
    std = 0.5

    normalized = output_target_norm(
        output_arr=raw,
        target_max=10.0,
        target_min=0.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )
    recovered = denormalize_model_target(
        data=normalized,
        min_val=0.0,
        max_val=10.0,
        out_norm="log_standard",
        target_log_mean=mean,
        target_log_std=std,
    )

    np.testing.assert_allclose(recovered, raw, rtol=1e-6, atol=1e-6)


def test_log_standard_target_transform_requires_stats():
    raw = np.array([[1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="target_log_mean"):
        output_target_norm(output_arr=raw, target_max=1.0, target_min=0.0, out_norm="log_standard")


def test_get_hexel_binary_maps_respects_masked_arrays(recwarn):
    pred = np.ma.array(
        [[1.0, 5.0], [100.0, 2.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )
    gt = np.ma.array(
        [[1.0, 4.0], [9.0, 3.0]],
        mask=[[False, False], [True, False]],
        dtype=np.float32,
    )

    pred_bin, gt_bin = get_hexel_binary_maps(pred_grid=pred, gt_grid=gt, percentile=0.5)

    assert not recwarn
    assert not pred_bin[1, 0]
    assert pred_bin[0, 1]
    assert pred_bin[1, 1]
    assert not gt_bin[1, 0]
    assert gt_bin[0, 1]
    assert gt_bin[1, 1]
