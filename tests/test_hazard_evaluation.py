import numpy as np
import pytest

from src.datasets.postprocessing.evaluate_hazard import (
    classify_hazard_13,
    compute_raw_hazard,
    config_from_args,
    load_hazard_eval_config,
    parse_fi_cap_value,
    scale_hazard,
    summarize_classes,
    summarize_fi_cap_diagnostics,
)


def test_hazard_classification_matches_nrcan_thresholds():
    scaled = np.array(
        [
            np.nan,
            0.0,
            0.009,
            0.01,
            0.024,
            0.025,
            0.05,
            0.1,
            0.25,
            0.5,
            1.0,
            2.5,
            5.0,
            10.0,
            25.0,
            50.0,
        ],
        dtype=np.float32,
    )

    classes = classify_hazard_13(scaled)

    assert np.isnan(classes[0])
    np.testing.assert_array_equal(classes[1:].astype(int), np.array([1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]))


def test_raw_hazard_caps_fi_and_multiplies_by_bp():
    bp = np.array([[0.5, 0.25, 2.0, np.nan]], dtype=np.float32)
    fi = np.array([[20_000.0, 400.0, 10.0, 1000.0]], dtype=np.float32)

    raw = compute_raw_hazard(bp, fi)

    np.testing.assert_allclose(raw[0, :3], np.array([5000.0, 100.0, 10.0], dtype=np.float32))
    assert np.isnan(raw[0, 3])


def test_raw_hazard_can_run_uncapped_for_sensitivity():
    bp = np.array([[0.5]], dtype=np.float32)
    fi = np.array([[20_000.0]], dtype=np.float32)

    raw = compute_raw_hazard(bp, fi, fi_cap=None)

    np.testing.assert_allclose(raw, np.array([[10_000.0]], dtype=np.float32))


def test_fi_cap_diagnostics_report_target_pred_clipping():
    target_fi = np.array([100.0, 20_000.0, 30_000.0, np.nan], dtype=np.float32)
    pred_fi = np.array([200.0, 5_000.0, 25_000.0, 1_000.0], dtype=np.float32)

    metrics = summarize_fi_cap_diagnostics(pred_fi, target_fi, fi_cap=10_000.0)

    assert metrics["fi_pair_count"] == pytest.approx(3.0)
    assert metrics["target_fi_gt_cap_frac"] == pytest.approx(2.0 / 3.0)
    assert metrics["pred_fi_gt_cap_frac"] == pytest.approx(1.0 / 3.0)
    assert metrics["target_fi_max"] == pytest.approx(30_000.0)
    assert metrics["pred_fi_max"] == pytest.approx(25_000.0)


def test_scale_hazard_uses_supplied_product_max():
    raw = np.array([0.0, 2.0, 4.0], dtype=np.float32)

    scaled = scale_hazard(raw, hazard_max=4.0)

    np.testing.assert_allclose(scaled, np.array([0.0, 50.0, 100.0], dtype=np.float32))


def test_class_metrics_include_within_k_and_high_iou():
    target = np.array([1, 10, 11, 13, np.nan], dtype=np.float32)
    pred = np.array([1, 11, 12, 13, 13], dtype=np.float32)

    metrics = summarize_classes(pred, target, high_class_thresholds=(11,))

    assert metrics["class_count"] == pytest.approx(4.0)
    assert metrics["class_accuracy"] == pytest.approx(0.5)
    assert metrics["class_within_1_accuracy"] == pytest.approx(1.0)
    assert metrics["class_mae"] == pytest.approx(0.5)
    assert metrics["class_ge_11_iou"] == pytest.approx(2.0 / 3.0)


def test_load_hazard_eval_config_reads_yaml(tmp_path):
    config_path = tmp_path / "hazard.yaml"
    config_path.write_text(
        """
raw_data_dir: /raw
bp_pred_dir: /bp
fi_pred_dir: /fi
hex_ids: [1, "12"]
mask_scope: actual
save_dir: /out
spearman_sample_size: 123
seed: 7
fi_cap_sensitivity: [10000, none, 5000]
""".strip()
    )

    config = load_hazard_eval_config(config_path)

    assert str(config.raw_data_dir) == "/raw"
    assert str(config.bp_pred_dir) == "/bp"
    assert str(config.fi_pred_dir) == "/fi"
    assert config.hex_ids == ["01", "12"]
    assert config.mask_scope == "actual"
    assert str(config.save_dir) == "/out"
    assert config.spearman_sample_size == 123
    assert config.seed == 7
    assert config.fi_caps == (10_000.0, None, 5_000.0)


def test_parse_fi_cap_value_accepts_none_aliases():
    assert parse_fi_cap_value("none") is None
    assert parse_fi_cap_value("uncapped") is None
    assert parse_fi_cap_value("20000") == pytest.approx(20_000.0)


def test_config_from_args_requires_cli_values_without_config():
    class Args:
        config = None
        raw_data_dir = None
        bp_pred_dir = None
        fi_pred_dir = None
        hex_ids = None
        mask_scope = "actual"
        save_dir = None
        spearman_sample_size = 10
        seed = 42
        fi_cap = "10000"
        fi_cap_sensitivity = None

    with pytest.raises(ValueError, match="Missing required CLI arguments"):
        config_from_args(Args())
