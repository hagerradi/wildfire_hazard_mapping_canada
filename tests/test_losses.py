import pytest
import torch

from src.losses import BCELoss, MSELoss


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    targets = torch.rand(4, 1, 32, 32)
    preds = targets * 0.9
    masks = torch.rand(4, 1, 32, 32) > 0.5
    return preds, targets, masks


def test_bce_loss_no_mask(dummy_data):
    loss_fn = BCELoss()
    preds, targets, _ = dummy_data
    # Compute expected using BCEWithLogitsLoss
    expected = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction="mean")
    result = loss_fn(preds, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bce_loss_with_mask(dummy_data):
    loss_fn = BCELoss()
    preds, targets, masks = dummy_data
    # Compute expected masked BCE loss using PyTorch, but mask after reduction="none"
    bce = torch.nn.functional.binary_cross_entropy_with_logits(preds, targets, reduction="none")
    masked_loss = (bce * masks).sum() / masks.sum()

    result = loss_fn(preds, targets, masks)
    assert torch.allclose(result, masked_loss, atol=1e-6)


def test_bce_loss_all_masked(dummy_data):
    loss_fn = BCELoss()
    preds, _, _ = dummy_data
    targets = torch.zeros_like(preds)
    mask = torch.zeros_like(preds)
    result = loss_fn(preds, targets, mask)
    # Should not be nan or inf due to eps
    assert torch.isfinite(result)


def test_mse_loss_no_mask(dummy_data):
    preds, targets, _ = dummy_data
    loss_fn = MSELoss()
    expected = ((preds - targets) ** 2).mean()
    result = loss_fn(preds, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_with_mask(dummy_data):
    preds, targets, masks = dummy_data
    loss_fn = MSELoss()
    # Compute expected masked MSE manually
    loss = (preds - targets) ** 2
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(preds, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_all_masked(dummy_data):
    loss_fn = MSELoss()
    preds, _, _ = dummy_data
    targets = torch.zeros_like(preds)
    mask = torch.zeros_like(preds)
    result = loss_fn(preds, targets, mask)
    # Should not be nan or inf due to eps
    assert torch.isfinite(result)
