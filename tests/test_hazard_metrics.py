import numpy as np
import pytest

from src.datasets.postprocessing.hazard_metrics import (
    calculate_hazard_class_metrics,
    flatten_hazard_class_metrics,
)


class TestCalculateHazardClassMetrics:
    def test_perfect_prediction(self):
        gt = np.array([[1, 2, 3], [3, 2, 1]])
        m = calculate_hazard_class_metrics(gt, gt, num_classes=3)
        assert m["exact_accuracy"] == 1.0
        assert m["within_1_accuracy"] == 1.0
        assert m["mean_absolute_class_error"] == 0.0
        assert m["macro_iou"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_ordinal_accuracies_and_mae(self):
        gt = np.array([1, 1, 1, 1])
        pred = np.array([1, 2, 3, 4])
        m = calculate_hazard_class_metrics(pred, gt, num_classes=5)
        assert m["exact_accuracy"] == 0.25
        assert m["within_1_accuracy"] == 0.5
        assert m["within_2_accuracy"] == 0.75
        assert m["mean_absolute_class_error"] == pytest.approx((0 + 1 + 2 + 3) / 4)

    def test_confusion_matrix_orientation_rows_gt_cols_pred(self):
        gt = np.array([1, 1, 2])
        pred = np.array([2, 2, 2])
        m = calculate_hazard_class_metrics(pred, gt, num_classes=2)
        cm = m["confusion_matrix"]
        # rows = GT, cols = pred: two GT=1 predicted as 2, one GT=2 predicted as 2
        assert cm[0, 1] == 2
        assert cm[1, 1] == 1
        assert cm[0, 0] == 0

    def test_per_class_iou_f1_nan_for_absent_classes(self):
        gt = np.array([1, 1, 2])
        pred = np.array([1, 1, 2])
        m = calculate_hazard_class_metrics(pred, gt, num_classes=4)
        iou = m["per_class_iou"]
        f1 = m["per_class_f1"]
        assert iou[0] == 1.0
        assert iou[1] == 1.0
        assert np.isnan(iou[2])
        assert np.isnan(iou[3])
        assert np.isnan(f1[2])
        assert np.isnan(f1[3])

    def test_invalid_and_nonfinite_pixels_ignored(self):
        gt = np.array([0, 1, 2, np.nan])
        pred = np.array([3, 1, 2, 2])
        m = calculate_hazard_class_metrics(pred, gt, invalid_class=0, num_classes=3)
        # only pixels 1 and 2 are valid, both correct
        assert m["exact_accuracy"] == 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="share a shape"):
            calculate_hazard_class_metrics(np.array([1, 2]), np.array([1, 2, 3]))

    def test_no_valid_overlap_raises(self):
        gt = np.array([0, 0])
        pred = np.array([1, 2])
        with pytest.raises(ValueError, match="no valid overlapping"):
            calculate_hazard_class_metrics(pred, gt, invalid_class=0, num_classes=3)


class TestFlattenHazardClassMetrics:
    def test_flatten_expands_arrays_and_drops_confusion(self):
        gt = np.array([1, 2])
        m = calculate_hazard_class_metrics(gt, gt, num_classes=2)
        flat = flatten_hazard_class_metrics(m)
        assert "confusion_matrix" not in flat
        assert flat["exact_accuracy"] == 1.0
        assert "per_class_iou_1" in flat
        assert "per_class_iou_2" in flat
        assert all(isinstance(v, float) for v in flat.values())
