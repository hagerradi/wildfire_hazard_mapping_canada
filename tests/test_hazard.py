import numpy as np
import pytest

from src.datasets.postprocessing.hazard import (
    DEFAULT_HAZARD_BIN_THRESHOLDS,
    bin_scaled_hazard,
    cap_fire_intensity,
    compute_raw_hazard,
    max_finite_hazard,
    scale_hazard,
)


class TestCapFireIntensity:
    def test_caps_values_above_cap(self):
        fi = np.array([5.0, 10000.0, 25000.0])
        capped = cap_fire_intensity(fi, fi_cap=10000.0)
        np.testing.assert_array_equal(capped, np.array([5.0, 10000.0, 10000.0]))

    def test_uncapped_when_none(self):
        fi = np.array([5.0, 25000.0])
        capped = cap_fire_intensity(fi, fi_cap=None)
        np.testing.assert_array_equal(capped, fi)

    def test_preserves_nan(self):
        fi = np.array([np.nan, 20000.0])
        capped = cap_fire_intensity(fi, fi_cap=10000.0)
        assert np.isnan(capped[0])
        assert capped[1] == 10000.0

    def test_preserves_nan_uncapped(self):
        fi = np.array([np.nan, 20000.0])
        capped = cap_fire_intensity(fi, fi_cap=None)
        assert np.isnan(capped[0])

    @pytest.mark.parametrize("bad_cap", [0.0, -1.0])
    def test_rejects_non_positive_cap(self, bad_cap):
        with pytest.raises(ValueError, match="fi_cap must be a positive"):
            cap_fire_intensity(np.array([1.0]), fi_cap=bad_cap)


class TestComputeRawHazard:
    def test_formula(self):
        bp = np.array([0.5, 1.0])
        fi = np.array([4.0, 3.0])
        raw = compute_raw_hazard(bp, fi, fi_cap=None)
        np.testing.assert_array_equal(raw, np.array([2.0, 3.0]))

    def test_applies_cap(self):
        bp = np.array([2.0])
        fi = np.array([50000.0])
        raw = compute_raw_hazard(bp, fi, fi_cap=10000.0)
        np.testing.assert_array_equal(raw, np.array([20000.0]))

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError, match="identical shape"):
            compute_raw_hazard(np.zeros((2, 2)), np.zeros((2, 3)))

    def test_nan_propagation(self):
        bp = np.array([np.nan, 1.0, 0.5])
        fi = np.array([1.0, np.nan, 2.0])
        raw = compute_raw_hazard(bp, fi, fi_cap=None)
        assert np.isnan(raw[0])
        assert np.isnan(raw[1])
        assert raw[2] == 1.0

    def test_returns_float(self):
        raw = compute_raw_hazard(np.array([1]), np.array([2]))
        assert raw.dtype == np.float64


class TestScaleHazard:
    def test_formula(self):
        raw = np.array([2.0, 4.0])
        scaled = scale_hazard(raw, denominator=8.0, scale_to=100.0)
        np.testing.assert_allclose(scaled, np.array([25.0, 50.0]))

    def test_preserves_nan(self):
        raw = np.array([np.nan, 4.0])
        scaled = scale_hazard(raw, denominator=8.0, scale_to=100.0)
        assert np.isnan(scaled[0])

    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_rejects_bad_denominator(self, bad):
        with pytest.raises(ValueError, match="denominator must be a positive"):
            scale_hazard(np.array([1.0]), denominator=bad)

    @pytest.mark.parametrize("bad", [0.0, -3.0])
    def test_rejects_bad_scale_to(self, bad):
        with pytest.raises(ValueError, match="scale_to must be a positive"):
            scale_hazard(np.array([1.0]), denominator=1.0, scale_to=bad)


class TestBinScaledHazard:
    def test_below_first_threshold_is_class_1(self):
        scaled = np.array([0.0, 0.009])
        classes = bin_scaled_hazard(scaled)
        np.testing.assert_array_equal(classes, np.array([1, 1]))

    def test_value_at_threshold_moves_to_higher_class(self):
        # Each threshold value should land in the higher class.
        scaled = np.array(DEFAULT_HAZARD_BIN_THRESHOLDS, dtype=float)
        classes = bin_scaled_hazard(scaled)
        # thresholds[k] -> class k + 2 (1-based, boundary goes up)
        expected = np.arange(2, len(DEFAULT_HAZARD_BIN_THRESHOLDS) + 2)
        np.testing.assert_array_equal(classes, expected)

    def test_just_below_threshold_stays_lower(self):
        scaled = np.array([0.01 - 1e-9, 0.025 - 1e-9])
        classes = bin_scaled_hazard(scaled)
        np.testing.assert_array_equal(classes, np.array([1, 2]))

    def test_at_and_above_last_threshold_is_top_class(self):
        scaled = np.array([50.0, 123.0])
        classes = bin_scaled_hazard(scaled)
        np.testing.assert_array_equal(classes, np.array([13, 13]))

    def test_invalid_pixels_get_invalid_class(self):
        scaled = np.array([np.nan, np.inf, -np.inf, 100.0])
        classes = bin_scaled_hazard(scaled)
        np.testing.assert_array_equal(classes, np.array([0, 0, 0, 13]))

    def test_custom_invalid_class(self):
        scaled = np.array([np.nan, 100.0])
        classes = bin_scaled_hazard(scaled, invalid_class=-1)
        np.testing.assert_array_equal(classes, np.array([-1, 13]))

    def test_custom_thresholds(self):
        scaled = np.array([0.5, 1.0, 5.0])
        classes = bin_scaled_hazard(scaled, thresholds=[1.0, 2.0])
        np.testing.assert_array_equal(classes, np.array([1, 2, 3]))

    def test_integer_output(self):
        classes = bin_scaled_hazard(np.array([1.0]))
        assert np.issubdtype(classes.dtype, np.integer)

    @pytest.mark.parametrize(
        "bad",
        [[1.0, 1.0], [2.0, 1.0], [-1.0, 2.0], [0.0, 1.0], []],
    )
    def test_invalid_thresholds_raise(self, bad):
        with pytest.raises(ValueError, match="thresholds"):
            bin_scaled_hazard(np.array([1.0]), thresholds=bad)


class TestMaxFiniteHazard:
    def test_single_grid(self):
        grid = np.array([1.0, 5.0, 2.0])
        assert max_finite_hazard(grid) == 5.0

    def test_multiple_grids(self):
        a = np.array([1.0, 3.0])
        b = np.array([[7.0, 2.0], [4.0, 6.0]])
        assert max_finite_hazard(a, b) == 7.0

    def test_ignores_nan_and_inf(self):
        grid = np.array([np.nan, np.inf, 4.0])
        assert max_finite_hazard(grid) == 4.0

    def test_all_nan_raises(self):
        with pytest.raises(ValueError, match="no finite hazard values"):
            max_finite_hazard(np.array([np.nan, np.inf]))

    def test_all_zero_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            max_finite_hazard(np.array([0.0, 0.0]))

    def test_negative_max_raises(self):
        with pytest.raises(ValueError, match="must be > 0"):
            max_finite_hazard(np.array([-1.0, -5.0]))

    def test_no_grids_raises(self):
        with pytest.raises(ValueError, match="at least one grid"):
            max_finite_hazard()
