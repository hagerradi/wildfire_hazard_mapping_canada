import json

import numpy as np
import pandas as pd
import pytest
import torch

from src.config import (
    Config,
    DataConfig,
    DataPrepConfig,
    DataSourceConfig,
    EvaluationConfig,
    GridParams,
    LoggerConfig,
    ModelConfig,
    OptimizerConfig,
    TrainingConfig,
)
from src.datasets.postprocessing import utils as post_utils
from src.datasets.postprocessing.diagnose_dynamic_range import (
    fixed_threshold_binary_maps,
    normalize_threshold_quantiles,
    summarize_bin_ranking,
    summarize_dynamic_range,
    summarize_threshold_exceedance,
)
from src.datasets.postprocessing.diagnose_neighbor_overlap_consistency import (
    build_overlap_row,
    overlap_bbox,
    parse_pair,
    prediction_path,
    summarize_pair_values,
)
from src.datasets.postprocessing.diagnose_residuals import _edge_distance_codes, lagged_spatial_correlations, summarize_grouped_errors
from src.datasets.postprocessing.diagnose_target_bins import summarize_quantile_bins, summarize_topk_bins
from src.datasets.postprocessing.quantile_calibration import apply_quantile_mapping, fit_quantile_mapping
from src.datasets.postprocessing.stitch_hexel import compute_coverage_count, stitch_windows, stitch_windows_with_diagnostics
from src.datasets.postprocessing.utils import (
    denormalize_model_target,
    effective_robust_plot_percentile,
    get_hexel_binary_maps,
    get_stitched_windows,
)
from src.datasets.postprocessing.visualize_predictions import (
    _target_prediction_diff_grids,
    _valid_pair_values,
    _value_scale_values,
    get_distribution_axis_limit,
    plot_hexbin_distribution,
    plot_histogram_distribution,
    visualize_hexel_iou,
    visualize_target_grids,
)
from src.datasets.targets import get_target_spec
from src.datasets.utils import output_burn_prob_norm


def test_get_stitched_windows_uses_prediction_mask_when_provided(tmp_path):
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
        prediction_mask_channel_indices=[0],
        win_h=2,
        win_w=2,
    )

    assert stitched[0, 0] == 1.0
    assert stitched[0, 1] == 2.0
    assert stitched[1, 0] == 3.0
    assert stitched[1, 1] == 4.0


def test_get_stitched_windows_falls_back_to_target_mask(tmp_path):
    patch = np.ones((2, 2, 7), dtype=np.float32)
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


def test_bp_target_nodata_is_zero_inside_prediction_support():
    gt = np.array([[np.nan, 0.2], [np.nan, np.nan]], dtype=np.float32)
    pred = np.array([[0.1, 0.3], [np.nan, 0.4]], dtype=np.float32)

    filled_gt, filled_pred = post_utils.fill_bp_target_nodata_as_zero(
        gt_grid=gt,
        pred_grid=pred,
        target=get_target_spec("bp"),
    )

    np.testing.assert_allclose(filled_pred, pred, equal_nan=True)
    assert filled_gt[0, 0] == pytest.approx(0.0)
    assert filled_gt[0, 1] == pytest.approx(0.2)
    assert np.isnan(filled_gt[1, 0])
    assert filled_gt[1, 1] == pytest.approx(0.0)


def test_non_bp_target_nodata_is_not_zero_filled():
    gt = np.array([[np.nan, 2.0]], dtype=np.float32)
    pred = np.array([[1.0, 3.0]], dtype=np.float32)

    filled_gt, filled_pred = post_utils.fill_bp_target_nodata_as_zero(
        gt_grid=gt,
        pred_grid=pred,
        target=get_target_spec("fi"),
    )

    np.testing.assert_allclose(filled_pred, pred, equal_nan=True)
    np.testing.assert_allclose(filled_gt, gt, equal_nan=True)


def test_load_target_grid_bp_nodata_zero_fill_is_configurable(monkeypatch, tmp_path):
    class FakePaths:
        def output_burn_prob(self):
            return tmp_path / "bp.tif"

        def mask_grid(self, hex_id, mask_scope):
            return tmp_path / f"hex{hex_id}_{mask_scope}.shp"

        def mask_grid_actual(self, hex_id):
            return tmp_path / f"hex{hex_id}_actual.shp"

    monkeypatch.setattr(
        post_utils,
        "load_spatial_raster",
        lambda *args, **kwargs: (np.array([[np.nan, 0.2]], dtype=np.float32), {"dtype": "float32"}),
    )
    monkeypatch.setattr(
        post_utils,
        "apply_mask_scope_to_grids",
        lambda gt_grid, pred_grid, **kwargs: (gt_grid, pred_grid),
    )

    pred = np.array([[0.1, 0.3]], dtype=np.float32)
    no_fill_gt, _ = post_utils.load_target_grid_for_mask_scope(
        paths=FakePaths(),
        target=get_target_spec("bp"),
        pred_grid=pred,
        profile={},
        mask_scope="actual",
        hex_id="01",
        bp_nodata_as_zero=False,
    )
    fill_gt, _ = post_utils.load_target_grid_for_mask_scope(
        paths=FakePaths(),
        target=get_target_spec("bp"),
        pred_grid=pred,
        profile={},
        mask_scope="actual",
        hex_id="01",
        bp_nodata_as_zero=True,
    )

    assert np.isnan(no_fill_gt[0, 0])
    assert fill_gt[0, 0] == pytest.approx(0.0)


def test_prediction_support_policy_can_use_target_mask(tmp_path):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    params = GridParams(feature_names_list=["ignition_grid"], target_name="bp")

    assert (
        post_utils.get_prediction_mask_channel_indices(
            data_dir=str(tmp_path),
            modelling_approach="1",
            grid_params=params,
            prediction_support_policy="target",
        )
        is None
    )
    assert post_utils.get_prediction_mask_channel_indices(
        data_dir=str(tmp_path),
        modelling_approach="1",
        grid_params=params,
        prediction_support_policy="input",
    ) == [0]


def test_evaluate_and_visualize_hexels_hides_support_outline_for_target_policy(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(
            best_ckpt_metrics=["loss"],
            best_ckpt_metrics_mode=["min"],
            prediction_support_policy="target",
        ),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    visualize_calls = []

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", lambda *args, **kwargs: (np.zeros((2, 2), dtype=np.float32), {}))
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: None)
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: visualize_calls.append(kwargs))
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions=None,
    )

    assert visualize_calls[0]["prediction_support_label"] == "target support"
    assert visualize_calls[0]["show_prediction_support_outline"] is False


def test_neighbor_overlap_summary_uses_finite_overlap_only():
    left = np.array([1.0, 3.0, np.nan, 7.0], dtype=np.float32)
    right = np.array([2.0, 1.0, 4.0, np.nan], dtype=np.float32)

    summary = summarize_pair_values(left, right, "pred")

    assert summary["pred_valid_pixels"] == 2.0
    assert summary["pred_bias"] == pytest.approx(0.5)
    assert summary["pred_mae"] == pytest.approx(1.5)
    assert summary["pred_rmse"] == pytest.approx(np.sqrt(2.5))
    assert parse_pair("hex01-hex5") == ("01", "05")


def test_neighbor_overlap_bbox_pads_valid_extent():
    mask = np.zeros((5, 6), dtype=bool)
    mask[2:4, 3:5] = True

    row_slice, col_slice = overlap_bbox(mask, pad=1)

    assert (row_slice.start, row_slice.stop) == (1, 5)
    assert (col_slice.start, col_slice.stop) == (2, 6)
    assert overlap_bbox(mask, pad=0) == (slice(2, 4), slice(3, 5))
    assert overlap_bbox(np.zeros((2, 2), dtype=bool)) is None


def test_neighbor_prediction_path_accepts_predicted_hexels_or_direct_dir(tmp_path):
    target = get_target_spec("bp")
    nested_dir = tmp_path / "nested"
    nested_pred_dir = nested_dir / "predicted_hexels"
    nested_pred_dir.mkdir(parents=True)
    nested_path = nested_pred_dir / "hexel_01_predicted.tif"
    nested_path.touch()

    direct_dir = tmp_path / "direct"
    direct_dir.mkdir()
    direct_path = direct_dir / "hexel_01_predicted.tif"
    direct_path.touch()

    assert prediction_path(nested_dir, "01", target) == nested_path
    assert prediction_path(direct_dir, "01", target) == direct_path


def test_neighbor_overlap_row_computes_difference_of_differences():
    left_pred = np.array([[0.7, 0.4], [0.2, np.nan]], dtype=np.float32)
    right_pred = np.array([[0.5, 0.1], [0.3, 0.2]], dtype=np.float32)
    left_target = np.array([[0.6, 0.2], [0.4, 0.1]], dtype=np.float32)
    right_target = np.array([[0.4, 0.3], [0.1, 0.2]], dtype=np.float32)

    arrays, summary = build_overlap_row(
        left_hex="01",
        right_hex="05",
        target=get_target_spec("bp"),
        left_pred=left_pred,
        right_pred=right_pred,
        left_target=left_target,
        right_target=right_target,
        max_plot_dim=10,
    )

    expected_target_diff = np.array([[0.2, -0.1], [0.3, np.nan]], dtype=np.float32)
    expected_pred_diff = np.array([[0.2, 0.3], [-0.1, np.nan]], dtype=np.float32)
    expected_diff_of_diff = expected_pred_diff - expected_target_diff
    np.testing.assert_allclose(arrays["target_diff"], expected_target_diff, equal_nan=True, atol=1e-7)
    np.testing.assert_allclose(arrays["pred_diff"], expected_pred_diff, equal_nan=True, atol=1e-7)
    np.testing.assert_allclose(arrays["diff_of_diff"], expected_diff_of_diff, equal_nan=True, atol=1e-7)
    assert summary["diff_of_diff_mae"] == pytest.approx(np.nanmean(np.abs(expected_diff_of_diff)))


def test_summarize_quantile_bins_reports_bias_by_target_bin():
    target = np.array([0.0, 2.0, 4.0, 10.0], dtype=np.float32)
    pred = np.array([0.0, 1.0, 3.0, 8.0], dtype=np.float32)

    summary = summarize_quantile_bins(pred, target, quantile_edges=(0.0, 0.5, 1.0))

    assert list(summary["bin"]) == ["q0.000-0.500", "q0.500-1.000"]
    assert summary.loc[0, "count"] == pytest.approx(2.0)
    assert summary.loc[0, "target_mean"] == pytest.approx(1.0)
    assert summary.loc[0, "pred_mean"] == pytest.approx(0.5)
    assert summary.loc[0, "bias"] == pytest.approx(-0.5)
    assert summary.loc[1, "count"] == pytest.approx(2.0)
    assert summary.loc[1, "target_mean"] == pytest.approx(7.0)
    assert summary.loc[1, "pred_mean"] == pytest.approx(5.5)
    assert summary.loc[1, "bias"] == pytest.approx(-1.5)


def test_summarize_quantile_bins_accepts_shared_thresholds():
    target = np.array([0.0, 2.0, 4.0], dtype=np.float32)
    pred = np.array([0.0, 1.0, 3.0], dtype=np.float32)

    summary = summarize_quantile_bins(
        pred,
        target,
        quantile_edges=(0.0, 0.5, 1.0),
        thresholds=np.array([0.0, 5.0, 10.0], dtype=np.float32),
    )

    assert summary.loc[0, "count"] == pytest.approx(3.0)
    assert summary.loc[0, "target_min"] == pytest.approx(0.0)
    assert summary.loc[0, "target_max"] == pytest.approx(5.0)
    assert summary.loc[1, "count"] == pytest.approx(0.0)


def test_summarize_topk_bins_reports_cumulative_hotspot_bias():
    target = np.array([0.0, 2.0, 4.0, 10.0], dtype=np.float32)
    pred = np.array([0.0, 1.0, 3.0, 8.0], dtype=np.float32)

    summary = summarize_topk_bins(pred, target, top_fractions=(0.5, 0.25))

    assert list(summary["bin"]) == ["top0.500", "top0.250"]
    assert summary.loc[0, "count"] == pytest.approx(2.0)
    assert summary.loc[0, "target_mean"] == pytest.approx(7.0)
    assert summary.loc[0, "bias"] == pytest.approx(-1.5)
    assert summary.loc[1, "count"] == pytest.approx(1.0)
    assert summary.loc[1, "target_mean"] == pytest.approx(10.0)
    assert summary.loc[1, "bias"] == pytest.approx(-2.0)


def test_quantile_mapping_corrects_monotonic_bias():
    pred = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    target = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)

    mapping = fit_quantile_mapping(pred, target, quantiles=(0.0, 0.5, 1.0))
    calibrated = apply_quantile_mapping(np.array([0.5, 2.5], dtype=np.float32), mapping)

    np.testing.assert_allclose(calibrated, np.array([1.0, 5.0], dtype=np.float32))


def test_quantile_mapping_handles_duplicate_prediction_knots_and_preserves_nan():
    pred = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    target = np.array([0.0, 1.0, 2.0, 4.0], dtype=np.float32)

    mapping = fit_quantile_mapping(pred, target, quantiles=(0.0, 0.25, 0.5, 0.75, 1.0), output_min=0.0, output_max=3.0)
    calibrated = apply_quantile_mapping(np.array([np.nan, -1.0, 0.0, 1.0, 2.0], dtype=np.float32), mapping)

    assert np.isnan(calibrated[0])
    assert calibrated[1] >= 0.0
    assert calibrated[-1] <= 3.0
    assert mapping.pred_knots.size < 5


def test_summarize_dynamic_range_reports_mae_relative_to_iqr():
    target = np.array([0.0, 2.0, 4.0, 6.0], dtype=np.float32)
    pred = np.array([0.0, 1.0, 3.0, 4.0], dtype=np.float32)

    summary = summarize_dynamic_range(pred, target)

    assert summary["count"] == pytest.approx(4.0)
    assert summary["mae"] == pytest.approx(1.0)
    assert summary["target_iqr"] == pytest.approx(3.0)
    assert summary["pred_iqr"] == pytest.approx(2.5)
    assert summary["mae_target_iqr_ratio"] == pytest.approx(1.0 / 3.0)
    assert summary["pred_target_iqr_ratio"] == pytest.approx(2.5 / 3.0)


def test_summarize_bin_ranking_detects_coarse_ordering():
    target = np.arange(10, dtype=np.float32)
    pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4], dtype=np.float32)

    summary, details = summarize_bin_ranking(pred, target, bin_count=5)

    assert summary["actual_bin_count"] == pytest.approx(5.0)
    assert summary["bin_mean_spearman"] == pytest.approx(1.0)
    assert summary["bin_pair_order_accuracy"] == pytest.approx(1.0)
    assert list(details["pred_mean"]) == pytest.approx([0.0, 1.0, 2.0, 3.0, 4.0])


def test_summarize_threshold_exceedance_uses_fixed_threshold_area():
    target = np.array([0.0, 1.0, 2.0, 3.0], dtype=np.float32)
    pred = np.array([0.0, 2.5, 2.5, 2.5], dtype=np.float32)

    summary = summarize_threshold_exceedance(pred, target, thresholds={0.75: 2.0})

    assert summary.loc[0, "target_count"] == pytest.approx(2.0)
    assert summary.loc[0, "pred_count"] == pytest.approx(3.0)
    assert summary.loc[0, "pred_target_area_ratio"] == pytest.approx(1.5)
    assert summary.loc[0, "recall"] == pytest.approx(1.0)
    assert summary.loc[0, "precision"] == pytest.approx(2.0 / 3.0)


def test_fixed_threshold_binary_maps_uses_same_absolute_threshold():
    target = np.array([[0.0, 1.0], [2.0, np.nan]], dtype=np.float32)
    pred = np.array([[0.5, 2.5], [1.5, 3.0]], dtype=np.float32)

    pred_bin, target_bin, valid_mask = fixed_threshold_binary_maps(pred, target, threshold=2.0)

    np.testing.assert_array_equal(valid_mask, np.array([[True, True], [True, False]]))
    np.testing.assert_array_equal(pred_bin, np.array([[False, True], [False, False]]))
    np.testing.assert_array_equal(target_bin, np.array([[False, False], [True, False]]))


def test_normalize_threshold_quantiles_accepts_fraction_and_percent_forms():
    assert normalize_threshold_quantiles([0.9, 95.0]) == pytest.approx((0.9, 0.95))

    with pytest.raises(ValueError, match="threshold quantiles"):
        normalize_threshold_quantiles([0.0])


def test_evaluate_and_visualize_hexels_stitches_each_multitarget_channel(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump(
            {
                "ignition_grid": [0],
                "bp_out_grid": [3],
                "fi_out_grid": [4],
                "ros_out_grid": [5],
            },
            f,
        )

    patch = np.ones((2, 2, 6), dtype=np.float32)
    patch[:, :, 3] = 1.0
    patch[0, 1, 3] = np.nan
    patch[:, :, 4] = 1.0
    patch[:, :, 5] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=3, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="multi_target", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[
                DataSourceConfig(
                    name="grid",
                    params=GridParams(
                        feature_names_list=["ignition_grid"],
                        target_name=["bp", "fi", "ros"],
                        target_out_norms={"bp": "none", "fi": "none", "ros": "none"},
                    ),
                )
            ],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    saved_hexels = {}

    def fake_save_predicted_hexels(predicted_hexel, hexel_profile, hex_id, save_dir, target_name=None):
        saved_hexels[target_name] = np.asarray(predicted_hexel)

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    def mean_prediction(preds, targets, masks):
        return preds[masks].mean()

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", fake_save_predicted_hexels)
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    predictions = np.array(
        [
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[10.0, 20.0], [30.0, 40.0]],
                [[100.0, 200.0], [300.0, 400.0]],
            ]
        ],
        dtype=np.float32,
    )

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=predictions,
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mean_prediction": mean_prediction},
    )

    assert set(saved_hexels) == {"bp", "fi", "ros"}
    np.testing.assert_allclose(saved_hexels["bp"], np.array([[1.0, 2.0], [3.0, 4.0]]))
    np.testing.assert_allclose(saved_hexels["fi"], np.array([[10.0, 20.0], [30.0, 40.0]]))
    np.testing.assert_allclose(saved_hexels["ros"], np.array([[100.0, 200.0], [300.0, 400.0]]))
    assert metrics["hex01/bp_mean_prediction"] == pytest.approx(2.5)
    assert metrics["hex01/fi_mean_prediction"] == pytest.approx(25.0)
    assert metrics["hex01/ros_mean_prediction"] == pytest.approx(250.0)
    assert metrics["all/bp_mean_prediction"] == pytest.approx(2.5)


def test_evaluate_and_visualize_hexels_metrics_only_skips_artifacts(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    artifact_calls = []

    def fail_artifact_call(*args, **kwargs):
        artifact_calls.append((args, kwargs))

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", fail_artifact_call)
    monkeypatch.setattr(post_utils, "visualize_target_grids", fail_artifact_call)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", fail_artifact_call)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", fail_artifact_call)
    monkeypatch.setattr(post_utils, "visualize_hexel_iou", fail_artifact_call)

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        save_artifacts=False,
    )

    assert artifact_calls == []
    assert metrics["hex01/mae"] == pytest.approx(1.0)


def test_validate_patch_metadata_mask_scope_rejects_unmarked_buffer_data():
    with pytest.raises(ValueError, match="matching 'mask_scope'"):
        post_utils.validate_patch_metadata_mask_scope(pd.DataFrame({"hex_id": [1]}), "buffer")

    assert post_utils.validate_patch_metadata_mask_scope(pd.DataFrame({"mask_scope": ["buffer"]}), "buffer_only") == "buffer_only"


def test_evaluate_and_visualize_hexels_uses_buffer_scope_paths_and_outputs(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame(
        [
            {
                "filename": "patch.npy",
                "hex_id": 1,
                "valid_ratio": 1.0,
                "season": "spring",
                "cause": "H",
                "row": 0,
                "col": 0,
                "mask_scope": "buffer",
            }
        ]
    ).to_csv(tmp_path / "test_indices.csv", index=False)

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    mask_paths = []
    save_dirs = []

    def fake_load_spatial_raster(*args, **kwargs):
        mask_paths.append(str(kwargs["mask_path"]))
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: save_dirs.append(kwargs.get("save_dir", args[3])))
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        mask_scope="buffer",
    )

    assert all(path.endswith("hex01_buffer.shp") for path in mask_paths)
    assert save_dirs == [str(tmp_path / "out" / "buffer_mask_eval")]
    assert metrics["hex01/buffer_mae"] == pytest.approx(1.0)
    assert metrics["all/buffer_mae"] == pytest.approx(1.0)


def test_buffer_scope_evaluation_reports_actual_and_buffer_only_splits(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame(
        [
            {
                "filename": "patch.npy",
                "hex_id": 1,
                "valid_ratio": 1.0,
                "season": "spring",
                "cause": "H",
                "row": 0,
                "col": 0,
                "mask_scope": "buffer",
            }
        ]
    ).to_csv(tmp_path / "test_indices.csv", index=False)

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[DataSourceConfig(name="grid", params=GridParams(feature_names_list=["ignition_grid"], target_name="bp"))],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999, "crs": "EPSG:3978", "transform": "mock"}

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "_actual_area_mask", lambda mask_path, profile, shape: np.array([[True, False], [False, False]]))

    metrics = post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions={"mae": lambda preds, targets, masks: torch.mean(torch.abs(preds[masks] - targets[masks]))},
        mask_scope="buffer",
        save_artifacts=False,
    )

    assert metrics["hex01/buffer_mae"] == pytest.approx(1.0)
    assert metrics["hex01/actual_mae"] == pytest.approx(1.0)
    assert metrics["hex01/buffer_only_mae"] == pytest.approx(1.0)
    assert metrics["all/actual_mae"] == pytest.approx(1.0)
    assert metrics["all/buffer_only_mae"] == pytest.approx(1.0)


def test_apply_mask_scope_to_grids_masks_actual_area_for_buffer_only(monkeypatch, tmp_path):
    monkeypatch.setattr(
        post_utils,
        "_actual_area_mask",
        lambda mask_path, profile, shape: np.array([[True, False], [False, False]]),
    )

    gt_grid, pred_grid = post_utils.apply_mask_scope_to_grids(
        gt_grid=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        pred_grid=np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32),
        profile={},
        mask_path=tmp_path / "actual.shp",
        mask_scope="buffer_only",
        hex_id="01",
    )

    assert np.isnan(gt_grid[0, 0])
    assert np.isnan(pred_grid[0, 0])
    np.testing.assert_allclose(gt_grid[[0, 1, 1], [1, 0, 1]], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_allclose(pred_grid[[0, 1, 1], [1, 0, 1]], np.array([6.0, 7.0, 8.0]))


def test_evaluate_and_visualize_hexels_writes_optional_robust_plot(tmp_path, monkeypatch):
    with (tmp_path / "feature_channel_map_1.json").open("w") as f:
        json.dump({"ignition_grid": [0], "bp_out_grid": [3]}, f)

    patch = np.ones((2, 2, 4), dtype=np.float32)
    patch[:, :, 3] = 1.0
    np.save(tmp_path / "patch.npy", patch)

    pd.DataFrame([{"filename": "patch.npy", "hex_id": 1, "valid_ratio": 1.0, "season": "spring", "cause": "H", "row": 0, "col": 0}]).to_csv(
        tmp_path / "test_indices.csv", index=False
    )

    config = Config(
        save_dir=str(tmp_path / "out"),
        modelling_approach="1",
        model=ModelConfig(num_classes=1, input_branches=["spatial"], hidden_features=[8, 16]),
        optimizer=OptimizerConfig(loss="mse", name="Adam", lr=0.001),
        training=TrainingConfig(max_epochs=1, log_every_n_epoch=1),
        evaluation=EvaluationConfig(best_ckpt_metrics=["loss"], best_ckpt_metrics_mode=["min"]),
        data=DataConfig(
            root_dir=str(tmp_path),
            raw_data_dir=str(tmp_path),
            train_split="train_indices.csv",
            val_split="val_indices.csv",
            test_split="test_indices.csv",
            input_sources=[
                DataSourceConfig(
                    name="grid",
                    params=GridParams(feature_names_list=["ignition_grid"], target_name="bp", out_norm="none"),
                )
            ],
        ),
        logger=LoggerConfig(enabled=False, project_name="test", workspace="test", experiment_name="test"),
        metrics=["mae"],
        data_prep=DataPrepConfig(win_h=2, win_w=2),
    )

    visualize_calls = []

    def fake_load_spatial_raster(*args, **kwargs):
        return np.zeros((2, 2), dtype=np.float32), {"dtype": "float32", "nodata": -9999}

    monkeypatch.setattr(post_utils, "get_range_output", lambda *args, **kwargs: (1.0, 0.0))
    monkeypatch.setattr(post_utils, "load_spatial_raster", fake_load_spatial_raster)
    monkeypatch.setattr(post_utils, "save_predicted_hexels", lambda *args, **kwargs: None)
    monkeypatch.setattr(post_utils, "visualize_target_grids", lambda **kwargs: visualize_calls.append(kwargs))
    monkeypatch.setattr(post_utils, "plot_hexbin_distribution", lambda **kwargs: None)
    monkeypatch.setattr(post_utils, "plot_histogram_distribution", lambda **kwargs: None)

    post_utils.evaluate_and_visualize_hexels(
        test_predictions=np.ones((1, 1, 2, 2), dtype=np.float32),
        config=config,
        out_norm="none",
        device=torch.device("cpu"),
        metric_functions=None,
        robust_plot_percentile=99.0,
    )

    assert len(visualize_calls) == 2
    assert visualize_calls[0]["target_label"] == "Burn Probability"
    assert "value_percentile" not in visualize_calls[0]
    assert visualize_calls[1]["value_percentile"] == 99.0
    assert visualize_calls[1]["diff_percentile"] == 99.0
    assert visualize_calls[1]["filename_suffix"] == "_p99"


def test_compute_coverage_count_for_overlapping_windows():
    masks = [
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2), dtype=bool),
    ]
    coords = [(0, 0), (0, 1)]

    coverage = compute_coverage_count(coords=coords, masks=masks, original_shape=(2, 3))

    np.testing.assert_array_equal(
        coverage,
        np.array(
            [
                [1, 2, 1],
                [1, 2, 1],
            ],
            dtype=np.int32,
        ),
    )


def test_stitch_windows_with_diagnostics_reports_overlap_disagreement():
    windows = [
        np.ones((2, 2), dtype=np.float32),
        np.full((2, 2), 3.0, dtype=np.float32),
    ]
    masks = [
        np.ones((2, 2), dtype=bool),
        np.ones((2, 2), dtype=bool),
    ]
    coords = [(0, 0), (0, 1)]

    stitched, diagnostics = stitch_windows_with_diagnostics(
        windows=windows,
        coords=coords,
        masks=masks,
        original_shape=(2, 3),
    )

    np.testing.assert_allclose(
        stitched,
        np.array(
            [
                [1.0, 2.0, 3.0],
                [1.0, 2.0, 3.0],
            ]
        ),
    )
    np.testing.assert_allclose(diagnostics.overlap_range[:, 1], np.array([2.0, 2.0]))
    np.testing.assert_allclose(diagnostics.overlap_variance[:, 1], np.array([1.0, 1.0]))
    assert diagnostics.summary["valid_pixel_count"] == 6.0
    assert diagnostics.summary["multi_coverage_pixel_count"] == 2.0
    assert diagnostics.summary["max_coverage"] == 2.0


def test_center_crop_stitch_prefers_patch_centers_with_mean_fallback():
    windows = [
        np.ones((4, 4), dtype=np.float32),
        np.full((4, 4), 3.0, dtype=np.float32),
    ]
    masks = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]
    coords = [(0, 0), (0, 2)]

    stitched = stitch_windows(
        windows=windows,
        coords=coords,
        masks=masks,
        original_shape=(4, 6),
        mode="center_crop",
        center_crop_fraction=0.5,
    )

    assert stitched[1, 2] == 1.0
    assert stitched[1, 3] == 3.0
    assert stitched[0, 0] == 1.0
    assert stitched[0, 5] == 3.0


def test_non_overlap_stitch_uses_single_most_central_contribution():
    windows = [
        np.ones((4, 4), dtype=np.float32),
        np.full((4, 4), 3.0, dtype=np.float32),
    ]
    masks = [np.ones((4, 4), dtype=bool), np.ones((4, 4), dtype=bool)]
    coords = [(0, 0), (0, 2)]

    stitched = stitch_windows(
        windows=windows,
        coords=coords,
        masks=masks,
        original_shape=(4, 6),
        mode="non_overlap",
    )

    assert stitched[1, 2] == 1.0
    assert stitched[1, 3] == 3.0
    assert stitched[0, 0] == 1.0
    assert stitched[0, 5] == 3.0


def test_feathered_stitch_weights_patch_centers_more_than_edges():
    windows = [
        np.ones((3, 3), dtype=np.float32),
        np.full((3, 3), 3.0, dtype=np.float32),
    ]
    masks = [np.ones((3, 3), dtype=bool), np.ones((3, 3), dtype=bool)]
    coords = [(0, 0), (0, 1)]

    stitched = stitch_windows(windows=windows, coords=coords, masks=masks, original_shape=(3, 4), mode="feathered")

    assert 1.0 < stitched[1, 1] < 2.0
    assert 2.0 < stitched[1, 2] < 3.0


def test_stitch_diagnostics_reports_mean_patch_edge_distance():
    window = np.arange(9, dtype=np.float32).reshape(3, 3)

    _, diagnostics = stitch_windows_with_diagnostics(
        windows=[window],
        coords=[(0, 0)],
        masks=[np.ones((3, 3), dtype=bool)],
        original_shape=(3, 3),
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    np.testing.assert_allclose(diagnostics.mean_edge_distance, expected)
    assert diagnostics.summary["mean_edge_distance_valid"] == pytest.approx(1.0 / 9.0)


def test_edge_distance_codes_use_expected_bins():
    edge_distance = np.array([[0, 1, 2, 3, 4, 127, 128]], dtype=np.float32)

    codes = _edge_distance_codes(edge_distance)

    np.testing.assert_array_equal(codes, np.array([[0, 1, 2, 2, 3, 7, 8]], dtype=np.int16))


def test_summarize_grouped_errors_reports_physical_error_stats():
    diff = np.array([[1.0, -1.0, 3.0], [np.nan, 2.0, -2.0]], dtype=np.float32)
    groups = np.array([[1, 1, 2], [1, 2, 2]], dtype=np.float32)
    valid = np.isfinite(diff)

    summary = summarize_grouped_errors(
        diff=diff,
        valid_mask=valid,
        group_values=groups,
        group_name="toy_group",
        target="bp",
        hex_id="01",
    ).sort_values("group_value")

    group_1 = summary[summary["group_value"] == "1.0"].iloc[0]
    group_2 = summary[summary["group_value"] == "2.0"].iloc[0]
    assert group_1["count"] == 2.0
    assert group_1["mae"] == pytest.approx(1.0)
    assert group_1["bias"] == pytest.approx(0.0)
    assert group_2["count"] == 3.0
    assert group_2["mae"] == pytest.approx(7.0 / 3.0)
    assert group_2["bias"] == pytest.approx(1.0)


def test_lagged_spatial_correlations_reports_row_and_column_structure():
    values = np.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
        ],
        dtype=np.float32,
    )
    valid = np.ones_like(values, dtype=bool)

    summary = lagged_spatial_correlations(values=values, valid_mask=valid, lags=(1,))

    assert set(summary["axis"]) == {"row", "col"}
    assert set(summary["lag_px"]) == {1}
    assert summary["pair_count"].min() == 6.0
    assert summary["correlation"].min() == pytest.approx(1.0)


def test_lagged_spatial_correlations_respects_valid_mask():
    values = np.array([[1.0, 2.0, np.nan], [1.0, 2.0, 3.0]], dtype=np.float32)
    valid = np.isfinite(values)

    summary = lagged_spatial_correlations(values=values, valid_mask=valid, lags=(1,))
    row = summary[(summary["axis"] == "row") & (summary["lag_px"] == 1)].iloc[0]
    col = summary[(summary["axis"] == "col") & (summary["lag_px"] == 1)].iloc[0]

    assert row["pair_count"] == 2.0
    assert col["pair_count"] == 3.0


def test_distribution_axis_limit_uses_non_probability_fallback():
    empty = np.array([], dtype=np.float32)

    assert get_distribution_axis_limit(empty, empty, probability_scale=True) == 0.15
    assert get_distribution_axis_limit(empty, empty, probability_scale=False) == 1.0


def test_distribution_axis_limit_only_caps_probability_scale():
    gt_vals = np.array([0.0, 2.0], dtype=np.float32)
    pred_vals = np.array([3.0], dtype=np.float32)

    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=True) == 1.0
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) > 3.0


def test_distribution_plots_drop_masked_nodata_values(tmp_path):
    gt = np.ma.array(
        [[1.0, -9999.0], [3.0, 4.0]],
        mask=[[False, True], [False, False]],
        dtype=np.float32,
    )
    pred = np.array([[1.5, 9999.0], [2.5, 5.0]], dtype=np.float32)

    gt_vals, pred_vals = _valid_pair_values(gt_grid=gt, pred_grid=pred, hex_id="01")

    np.testing.assert_allclose(gt_vals, np.array([1.0, 3.0, 4.0], dtype=np.float32))
    np.testing.assert_allclose(pred_vals, np.array([1.5, 2.5, 5.0], dtype=np.float32))
    assert get_distribution_axis_limit(gt, pred, probability_scale=False) == pytest.approx(9999.0 * 1.05)
    assert get_distribution_axis_limit(gt_vals, pred_vals, probability_scale=False) == pytest.approx(5.0 * 1.05)

    plot_histogram_distribution(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), probability_scale=False)
    plot_hexbin_distribution(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), probability_scale=False)

    assert (tmp_path / "predicted_hexels_plot" / "hist_hex_01.png").exists()
    assert (tmp_path / "predicted_hexels_plot" / "hexbin_hex_01.png").exists()


def test_target_grid_visualization_handles_masked_nodata_and_constant_difference(tmp_path):
    gt = np.ma.array(
        [[2.0, -9999.0], [2.0, 2.0]],
        mask=[[False, True], [False, False]],
        dtype=np.float32,
    )
    pred = np.array([[2.0, 9999.0], [2.0, 2.0]], dtype=np.float32)

    visualize_target_grids(gt_grid=gt, pred_grid=pred, hex_id="01", save_dir=str(tmp_path), target_label="Fire Intensity")

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_target_grid_visualization_keeps_prediction_support_outside_target():
    gt = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    pred = np.array([[1.5, 2.5], [3.5, np.nan]], dtype=np.float32)

    gt_plot, pred_plot, diff_plot, pred_mask, overlap_mask = _target_prediction_diff_grids(gt_grid=gt, pred_grid=pred, hex_id="01")

    assert pred_plot[0, 1] == pytest.approx(2.5)
    assert np.isnan(gt_plot[0, 1])
    assert np.isnan(diff_plot[0, 1])
    assert np.isnan(pred_plot[1, 1])
    assert pred_mask.tolist() == [[True, True], [True, False]]
    assert overlap_mask.tolist() == [[True, False], [True, False]]


def test_target_grid_visualization_accepts_actual_boundary_mask(tmp_path):
    gt = np.array([[0.0, 1.0, np.nan], [0.0, 2.0, 3.0], [np.nan, 4.0, 5.0]], dtype=np.float32)
    pred = np.ones((3, 3), dtype=np.float32)
    actual_mask = np.array([[False, True, False], [True, True, True], [False, True, False]])

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Burn Probability",
        actual_support_mask=actual_mask,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_target_grid_visualization_accepts_buffer_boundary_mask(tmp_path):
    gt = np.array([[0.0, 1.0, np.nan], [0.0, 2.0, 3.0], [np.nan, 4.0, 5.0]], dtype=np.float32)
    pred = np.ones((3, 3), dtype=np.float32)
    buffer_mask = np.ones((3, 3), dtype=bool)

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Burn Probability",
        buffer_support_mask=buffer_mask,
        show_prediction_support_outline=False,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted.png").exists()


def test_iou_visualization_accepts_actual_and_buffer_boundary_masks(tmp_path):
    gt = np.array([[0.1, 0.2, np.nan], [0.3, 0.4, 0.5], [np.nan, 0.6, 0.7]], dtype=np.float32)
    pred = np.array([[0.2, 0.1, np.nan], [0.35, 0.45, 0.55], [np.nan, 0.65, 0.75]], dtype=np.float32)
    gt_bin = np.isfinite(gt) & (gt >= 0.5)
    pred_bin = np.isfinite(pred) & (pred >= 0.5)
    actual_mask = np.array([[False, True, False], [True, True, True], [False, True, False]])
    buffer_mask = np.isfinite(gt)

    visualize_hexel_iou(
        gt_grid=gt,
        pred_grid=pred,
        gt_bin=gt_bin,
        pred_bin=pred_bin,
        hex_id="01",
        save_dir=str(tmp_path),
        percentile=0.9,
        actual_support_mask=actual_mask,
        buffer_support_mask=buffer_mask,
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_top_10perc_iou.png").exists()


def test_target_grid_visualization_rejects_mismatched_actual_boundary_mask(tmp_path):
    gt = np.ones((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    actual_mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="actual_support_mask shape"):
        visualize_target_grids(
            gt_grid=gt,
            pred_grid=pred,
            hex_id="01",
            save_dir=str(tmp_path),
            actual_support_mask=actual_mask,
        )


def test_target_grid_visualization_rejects_mismatched_buffer_boundary_mask(tmp_path):
    gt = np.ones((2, 2), dtype=np.float32)
    pred = np.ones((2, 2), dtype=np.float32)
    buffer_mask = np.ones((3, 3), dtype=bool)

    with pytest.raises(ValueError, match="buffer_support_mask shape"):
        visualize_target_grids(
            gt_grid=gt,
            pred_grid=pred,
            hex_id="01",
            save_dir=str(tmp_path),
            buffer_support_mask=buffer_mask,
        )


def test_target_grid_visualization_writes_robust_percentile_variant(tmp_path):
    gt = np.array([[1.0, 2.0], [3.0, 1000.0]], dtype=np.float32)
    pred = np.array([[1.0, 1.5], [3.5, 10.0]], dtype=np.float32)

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Fire Intensity",
        value_percentile=99.0,
        diff_percentile=99.0,
        filename_suffix="_p99",
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted_p99.png").exists()


def test_fi_ros_default_to_p99_robust_plot_percentile():
    assert effective_robust_plot_percentile(get_target_spec("bp"), None) is None
    assert effective_robust_plot_percentile(get_target_spec("fi"), None) == 99.0
    assert effective_robust_plot_percentile(get_target_spec("ros"), None) == 99.0
    assert effective_robust_plot_percentile(get_target_spec("fi"), 98.0) == 98.0


def test_value_scale_values_can_use_target_only():
    gt = np.array([[0.0, 2.0], [np.nan, 10.0]], dtype=np.float32)
    pred = np.array([[1.0, 3.0], [4.0, np.nan]], dtype=np.float32)

    np.testing.assert_allclose(_value_scale_values(gt, pred, "target"), np.array([0.0, 2.0, 10.0], dtype=np.float32))
    np.testing.assert_allclose(_value_scale_values(gt, pred, "prediction"), np.array([1.0, 3.0, 4.0], dtype=np.float32))

    with pytest.raises(ValueError, match="value_percentile_source"):
        _value_scale_values(gt, pred, "bad")


def test_target_grid_visualization_accepts_target_percentile_source(tmp_path):
    gt = np.array([[0.0, 2.0], [4.0, 100.0]], dtype=np.float32)
    pred = np.array([[1.0, 1.5], [2.5, 3.0]], dtype=np.float32)

    visualize_target_grids(
        gt_grid=gt,
        pred_grid=pred,
        hex_id="01",
        save_dir=str(tmp_path),
        target_label="Burn Probability",
        value_percentile=90.0,
        value_percentile_source="target",
        filename_suffix="_gtp90",
    )

    assert (tmp_path / "predicted_hexels_plot" / "hexel_01_predicted_gtp90.png").exists()


def test_log_standard_target_transform_roundtrip():
    raw = np.array([[0.0, 1.0, 9.0]], dtype=np.float32)
    mean = 1.25
    std = 0.5

    normalized = output_burn_prob_norm(
        output_arr=raw,
        burn_prob_max=10.0,
        burn_prob_min=0.0,
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
        output_burn_prob_norm(output_arr=raw, burn_prob_max=1.0, burn_prob_min=0.0, out_norm="log_standard")


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
