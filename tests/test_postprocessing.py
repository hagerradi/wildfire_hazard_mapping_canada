import os

import numpy as np
import pandas as pd

from src.datasets.postprocessing.utils import get_stitched_windows
from src.datasets.postprocessing.visualize_predictions import get_distribution_axis_limit


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
