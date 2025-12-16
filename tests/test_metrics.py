import pytest
import torch

from src.metrics import compute_mae, compute_mse, compute_spearman, compute_ssim


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    targets = torch.rand(4, 1, 32, 32)
    preds = targets * 0.9
    return preds, targets


def test_mse_is_non_negative(dummy_data):
    preds, targets = dummy_data
    loss = compute_mse(preds, targets)
    assert loss >= 0
    assert isinstance(loss, torch.Tensor)


def test_mae_perfect_match():
    data = torch.ones(2, 1, 16, 16)
    loss = compute_mae(data, data)
    assert torch.isclose(loss, torch.tensor(0.0))


def test_spearman_perfect_correlation():
    targets = torch.rand(4, 1, 16, 16)
    preds = targets * 2.0
    score = compute_spearman(preds, targets)
    assert torch.isclose(score, torch.tensor(1.0), atol=1e-4)


def test_ssim_range(dummy_data):
    preds, targets = dummy_data
    score = compute_ssim(preds, targets)
    assert -1.0 <= score <= 1.0
