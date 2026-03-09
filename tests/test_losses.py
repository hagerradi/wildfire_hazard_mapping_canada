import pytest
import torch
import torch.nn.functional as F

from src.losses import (
    BCELoss,
    BernoulliKLLoss,
    CCCLoss,
    DiceLoss,
    FocalLoss,
    MAELoss,
    MSELoss,
    WeightedLoss,
)


@pytest.fixture
def dummy_data():
    torch.manual_seed(42)
    targets = torch.rand(4, 1, 32, 32)
    logits = targets * 0.9
    masks = torch.rand(4, 1, 32, 32) > 0.5
    return logits, targets, masks


# -------------------------
# BCE
# -------------------------


def test_bce_loss_no_mask(dummy_data):
    loss_fn = BCELoss()
    logits, targets, _ = dummy_data
    expected = F.binary_cross_entropy_with_logits(logits, targets, reduction="mean")
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bce_loss_with_mask(dummy_data):
    loss_fn = BCELoss()
    logits, targets, masks = dummy_data
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    masked_loss = (bce * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, masked_loss, atol=1e-6)


def test_bce_loss_all_masked(dummy_data):
    loss_fn = BCELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# MSE
# -------------------------


def test_mse_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = MSELoss()
    expected = ((torch.sigmoid(logits) - targets) ** 2).mean()
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = MSELoss()
    loss = (torch.sigmoid(logits) - targets) ** 2
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mse_loss_all_masked(dummy_data):
    loss_fn = MSELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# MAE
# -------------------------


def test_mae_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = MAELoss()
    expected = (torch.abs(torch.sigmoid(logits) - targets)).mean()
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mae_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = MAELoss()
    loss = torch.abs(torch.sigmoid(logits) - targets)
    expected = (loss * masks).sum() / masks.sum()
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_mae_loss_all_masked(dummy_data):
    loss_fn = MAELoss()
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Dice
# -------------------------


def _dice_reference(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None, eps: float):
    probs = torch.sigmoid(logits)
    if mask is None:
        probs_f = probs.flatten(1)
        targets_f = targets.flatten(1)
    else:
        m = mask.to(dtype=probs.dtype)
        probs_f = (probs * m).flatten(1)
        targets_f = (targets * m).flatten(1)

    intersection = (probs_f * targets_f).sum(dim=1)
    denom = probs_f.sum(dim=1) + targets_f.sum(dim=1)
    dice = (2.0 * intersection + eps) / (denom + eps)
    return 1.0 - dice.mean()


def test_dice_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = DiceLoss(eps=1e-8)
    expected = _dice_reference(logits, targets, None, eps=1e-8)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_dice_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = DiceLoss(eps=1e-8)
    expected = _dice_reference(logits, targets, masks, eps=1e-8)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_dice_loss_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = DiceLoss(eps=1e-8)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Focal
# -------------------------


def _focal_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    gamma: float,
    alpha: float | None,
    eps: float,
):
    targets = targets.to(dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

    probs = torch.sigmoid(logits)
    p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
    focal_factor = (1.0 - p_t).clamp_min(0.0).pow(gamma)

    if alpha is not None:
        alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
        loss = alpha_t * focal_factor * bce
    else:
        loss = focal_factor * bce

    if mask is None:
        return loss.mean()

    m = mask.to(dtype=loss.dtype)
    loss = loss * m
    denom = m.sum().clamp_min(eps)
    return loss.sum() / denom


@pytest.mark.parametrize("alpha", [0.25, None])
def test_focal_loss_no_mask_matches_reference(dummy_data, alpha):
    logits, targets, _ = dummy_data
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha, eps=1e-8)
    expected = _focal_reference(logits, targets, None, gamma=2.0, alpha=alpha, eps=1e-8)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


@pytest.mark.parametrize("alpha", [0.25, None])
def test_focal_loss_with_mask_matches_reference(dummy_data, alpha):
    logits, targets, masks = dummy_data
    loss_fn = FocalLoss(gamma=2.0, alpha=alpha, eps=1e-8)
    expected = _focal_reference(logits, targets, masks, gamma=2.0, alpha=alpha, eps=1e-8)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_focal_loss_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = FocalLoss(gamma=2.0, alpha=0.25, eps=1e-8)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# Bernoulli KL
# -------------------------


def _bernoulli_kl_reference(
    logits: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor | None,
    eps: float,
    clamp_logits: float | None,
):
    if clamp_logits is not None:
        logits = logits.clamp(-clamp_logits, clamp_logits)

    targets = targets.to(dtype=logits.dtype)
    p = targets.clamp(eps, 1.0 - eps)

    log_q = F.logsigmoid(logits)
    log_1mq = F.logsigmoid(-logits)
    log_p = torch.log(p)
    log_1mp = torch.log1p(-p)

    loss = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)

    if mask is None:
        return loss.mean()

    m = mask.to(dtype=loss.dtype)
    loss = torch.where(m > 0, loss, torch.zeros_like(loss))
    denom = m.sum().clamp_min(eps)
    return loss.sum() / denom


def test_bernoulli_kl_no_mask_matches_reference(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    expected = _bernoulli_kl_reference(logits, targets, None, eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bernoulli_kl_with_mask_matches_reference(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    expected = _bernoulli_kl_reference(logits, targets, masks, eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_bernoulli_kl_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = BernoulliKLLoss(eps=1e-6, clamp_logits=20.0)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# CCC
# -------------------------
def _ccc_reference(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None, eps: float):
    preds = torch.sigmoid(logits)
    if mask is None:
        fp = preds.flatten()
        ft = targets.flatten()
    else:
        valid = mask.bool().flatten()
        fp = preds.flatten()[valid]
        ft = targets.flatten()[valid]
    if fp.numel() == 0:
        return preds.new_tensor(1.0)
    mean_p = fp.mean()
    mean_t = ft.mean()
    var_p = fp.var(correction=0)
    var_t = ft.var(correction=0)
    cov_pt = ((fp - mean_p) * (ft - mean_t)).mean()
    ccc = 2.0 * cov_pt / (var_p + var_t + (mean_p - mean_t) ** 2 + eps)
    return 1.0 - ccc


def test_ccc_loss_no_mask(dummy_data):
    logits, targets, _ = dummy_data
    loss_fn = CCCLoss(eps=1e-8)
    expected = _ccc_reference(logits, targets, None, eps=1e-8)
    result = loss_fn(logits, targets)
    assert torch.allclose(result, expected, atol=1e-6)


def test_ccc_loss_with_mask(dummy_data):
    logits, targets, masks = dummy_data
    loss_fn = CCCLoss(eps=1e-8)
    expected = _ccc_reference(logits, targets, masks, eps=1e-8)
    result = loss_fn(logits, targets, masks)
    assert torch.allclose(result, expected, atol=1e-6)


def test_ccc_loss_all_masked_is_finite(dummy_data):
    logits, _, _ = dummy_data
    targets = torch.zeros_like(logits)
    mask = torch.zeros_like(logits)
    loss_fn = CCCLoss(eps=1e-8)
    result = loss_fn(logits, targets, mask)
    assert torch.isfinite(result)


# -------------------------
# WeightedLoss
# -------------------------


def test_weighted_loss_returns_total_and_parts(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 0.7, "mse": 0.3}
    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    total, parts = loss_fn(logits, targets, masks)

    assert isinstance(total, torch.Tensor)
    assert isinstance(parts, dict)
    assert set(parts.keys()) == {"bce", "mse"}


def test_weighted_loss_total_matches_manual(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 0.7, "mse": 0.3}

    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)
    total, parts = loss_fn(logits, targets, masks)

    # manual total using normalized weights
    w = torch.tensor([weights["bce"], weights["mse"]], dtype=torch.float32)
    w = w / w.sum().clamp_min(1e-8)

    expected_total = w[0].to(dtype=parts["bce"].dtype) * parts["bce"] + w[1].to(dtype=parts["mse"].dtype) * parts["mse"]
    assert torch.allclose(total, expected_total, atol=1e-6)


def test_weighted_loss_buffer_moves_device(dummy_data):
    logits, targets, masks = dummy_data
    losses = {"bce": BCELoss(), "mse": MSELoss()}
    weights = {"bce": 1.0, "mse": 1.0}
    loss_fn = WeightedLoss(losses=losses, weights=weights, normalize_weights=True)

    # Move to same device as logits (this is CPU in unit tests typically)
    loss_fn = loss_fn.to(logits.device)
    assert loss_fn._weights.device == logits.device
