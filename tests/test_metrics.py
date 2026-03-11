import pytest
import torch

from src.metrics import (
    compute_auc_iou,
    compute_bias,
    compute_ccc,
    compute_mae,
    compute_mse,
    compute_spearman,
    compute_ssim,
    compute_topK_iou,
)


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    targets = torch.rand(4, 1, 32, 32)
    preds = targets * 0.9
    return preds, targets


@pytest.fixture
def dummy_mask():
    torch.manual_seed(42)
    return (torch.rand(4, 1, 32, 32) > 0.5).float()


def test_mse_is_non_negative(dummy_data):
    preds, targets = dummy_data
    loss = compute_mse(preds, targets)
    assert loss >= 0
    assert isinstance(loss, torch.Tensor)


def test_mse_with_mask(dummy_data, dummy_mask):
    preds, targets = dummy_data
    loss = compute_mse(preds, targets, mask=dummy_mask)
    assert loss >= 0
    assert isinstance(loss, torch.Tensor)


def test_mae_perfect_match():
    data = torch.ones(2, 1, 16, 16)
    loss = compute_mae(data, data)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_mae_perfect_match_with_mask():
    data = torch.ones(2, 1, 16, 16)
    mask = torch.ones_like(data)
    loss = compute_mae(data, data, mask=mask)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_spearman_perfect_correlation():
    targets = torch.rand(4, 1, 16, 16)
    preds = targets * 2.0
    score = compute_spearman(preds, targets)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_spearman_perfect_correlation_with_mask():
    targets = torch.rand(4, 1, 16, 16)
    preds = targets * 2.0
    mask = torch.ones_like(targets)
    score = compute_spearman(preds, targets, mask=mask)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_ssim_range(dummy_data):
    preds, targets = dummy_data
    score = compute_ssim(preds, targets)
    assert -1.0 <= score <= 1.0


def test_ssim_range_with_mask(dummy_data, dummy_mask):
    preds, targets = dummy_data
    score = compute_ssim(preds, targets, mask=dummy_mask)
    assert -1.0 <= score <= 1.0


def test_topK_iou_perfect_match(dummy_data):
    _, targets = dummy_data
    iou = compute_topK_iou(targets, targets, percentile=0.90)
    assert torch.isclose(iou, torch.tensor(1.0))


def test_topK_iou_perfect_match_with_mask(dummy_data, dummy_mask):
    _, targets = dummy_data
    iou = compute_topK_iou(targets, targets, mask=dummy_mask, percentile=0.90)
    assert torch.isclose(iou, torch.tensor(1.0))


def test_topK_iou_range(dummy_data):
    preds, targets = dummy_data
    iou = compute_topK_iou(preds, targets, percentile=0.90)
    assert 0.0 <= iou.item() <= 1.0


def test_topK_iou_empty_mask_edge_case(dummy_data):
    preds, targets = dummy_data
    empty_mask = torch.zeros_like(targets)
    iou = compute_topK_iou(preds, targets, mask=empty_mask)
    assert torch.isnan(iou)


def test_topK_iou_completely_disjoint():
    targets = torch.zeros(1, 1, 1, 10)
    targets[..., -1] = 1.0

    preds = torch.zeros(1, 1, 1, 10)
    preds[..., 0] = 1.0

    iou = compute_topK_iou(preds, targets, percentile=0.90)
    assert torch.isclose(iou, torch.tensor(0.0))


def test_bias_zero_when_perfect_match(dummy_data, dummy_mask):
    _, targets = dummy_data
    bias = compute_bias(targets, targets, mask=dummy_mask)
    assert torch.isclose(bias, torch.tensor(0.0))


def test_bias_positive_and_negative_shifts(dummy_data, dummy_mask):
    _, targets = dummy_data
    preds_over = targets + 2.5
    bias_over = compute_bias(preds_over, targets, mask=dummy_mask)
    assert torch.isclose(bias_over, torch.tensor(2.5))

    preds_under = targets - 1.5
    bias_under = compute_bias(preds_under, targets, mask=dummy_mask)
    assert torch.isclose(bias_under, torch.tensor(-1.5))


def test_bias_ignores_masked_out_regions():
    targets = torch.zeros(1, 1, 4, 4)
    preds = torch.zeros(1, 1, 4, 4)
    mask = torch.zeros(1, 1, 4, 4)

    mask[..., :2, :2] = 1.0
    preds[..., :2, :2] = 3.0
    preds[..., 2:, 2:] = 100.0

    bias = compute_bias(preds, targets, mask=mask)
    assert torch.isclose(bias, torch.tensor(3.0))


def test_bias_empty_mask_edge_case(dummy_data):
    preds, targets = dummy_data
    empty_mask = torch.zeros_like(targets)
    bias = compute_bias(preds, targets, mask=empty_mask)
    assert torch.isclose(bias, torch.tensor(0.0))


def test_auc_iou_perfect_match(dummy_data):
    _, targets = dummy_data
    # test on top 10%
    auc = compute_auc_iou(targets, targets, k_values=(0.01, 0.10), steps=10)
    assert torch.isclose(auc, torch.tensor(1.0))


def test_auc_iou_perfect_match_with_mask(dummy_data, dummy_mask):
    _, targets = dummy_data
    # test on full range
    auc = compute_auc_iou(targets, targets, mask=dummy_mask, k_values=(0.01, 0.99), steps=50)
    assert torch.isclose(auc, torch.tensor(1.0))


def test_auc_iou_range(dummy_data):
    preds, targets = dummy_data
    auc = compute_auc_iou(preds, targets, k_values=(0.05, 0.20), steps=15)
    assert 0.0 <= auc.item() <= 1.0


def test_auc_iou_empty_mask_edge_case(dummy_data):
    preds, targets = dummy_data
    empty_mask = torch.zeros_like(targets)
    auc = compute_auc_iou(preds, targets, mask=empty_mask)
    assert torch.isnan(auc)


def test_auc_iou_completely_disjoint():
    targets = torch.zeros(1, 1, 1, 100)
    targets[..., -10:] = 1.0

    preds = torch.zeros(1, 1, 1, 100)
    preds[..., :10] = 1.0

    auc = compute_auc_iou(preds, targets, k_values=(0.01, 0.10), steps=10)
    assert torch.isclose(auc, torch.tensor(0.0))


def test_auc_iou_invalid_k_values(dummy_data):
    preds, targets = dummy_data

    with pytest.raises(ValueError):
        compute_auc_iou(preds, targets, k_values="all")  # type: ignore

    with pytest.raises(ValueError):
        compute_auc_iou(preds, targets, k_values=[0.01, 0.10])  # type: ignore


def test_ccc_perfect_agreement():
    targets = torch.rand(4, 1, 16, 16)
    ccc = compute_ccc(targets, targets)
    assert torch.isclose(ccc, torch.tensor(1.0), atol=1e-5)


def test_ccc_perfect_agreement_with_mask():
    targets = torch.rand(4, 1, 16, 16)
    mask = torch.ones_like(targets)
    ccc = compute_ccc(targets, targets, mask=mask)
    assert torch.isclose(ccc, torch.tensor(1.0), atol=1e-5)


def test_ccc_range(dummy_data):
    preds, targets = dummy_data
    ccc = compute_ccc(preds, targets)
    assert -1.0 <= ccc.item() <= 1.0


def test_ccc_range_with_mask(dummy_data, dummy_mask):
    preds, targets = dummy_data
    ccc = compute_ccc(preds, targets, mask=dummy_mask)
    assert -1.0 <= ccc.item() <= 1.0


def test_ccc_empty_mask_edge_case(dummy_data):
    preds, targets = dummy_data
    empty_mask = torch.zeros_like(targets)
    ccc = compute_ccc(preds, targets, mask=empty_mask)
    assert torch.isnan(ccc)


def test_ccc_scaled_preds_less_than_one():
    """CCC should be < 1 when preds are a scaled version of targets (not identical)."""
    targets = torch.rand(4, 1, 16, 16)
    preds = targets * 2.0
    ccc = compute_ccc(preds, targets)
    assert ccc.item() < 1.0
