# Definitions of loss functions
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class BCELoss(nn.Module):
    """
    Binary cross entropy loss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # internally handles sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.bce(logits, targets)  # same shape as logits/targets

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class MSELoss(nn.Module):
    """
    MSELoss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        loss = self.mse(probs, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom


class MAELoss(nn.Module):
    """
    MAELoss with optional mask
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        loss = self.mae(probs, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom


class HuberLoss(nn.Module):
    """
    Huber/SmoothL1 loss on raw model outputs with optional mask.

    This is intended for unconstrained regression targets, e.g. standardized log FI/ROS.
    """

    def __init__(self, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.huber = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.huber(logits, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class DiceLoss(nn.Module):
    """
    Soft Dice loss for segmentation.

    Expects:
      logits: shape (N, 1, H, W)
      targets: same shape, values in [0,1]
      mask: same shape (bool or 0/1), where 1 means valid pixel
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        probs = torch.sigmoid(logits)
        if mask is None:
            probs = probs.flatten(1)
            targets = targets.flatten(1)
            valid = None
        else:
            mask = mask.to(dtype=probs.dtype)
            valid = mask.flatten(1).sum(dim=1) > 0
            probs = (probs * mask).flatten(1)
            targets = (targets * mask).flatten(1)

        intersection = (probs * targets).sum(dim=1)
        denom = probs.sum(dim=1) + targets.sum(dim=1)
        dice = (2.0 * intersection + self.eps) / (denom + self.eps)

        if valid is None:
            return 1.0 - dice.mean()

        if valid.any():
            return 1.0 - dice[valid].mean()
        else:
            return dice.new_tensor(0.0)


class FocalLoss(nn.Module):
    """
    Binary focal loss

    Expects:
      logits: (N, 1, H, W)
      targets: same shape, values in [0,1]
      mask: same shape (bool or 0/1), where 1 means valid pixel

    Params:
      alpha: class balancing factor. Common: 0.25 for positives (as in RetinaNet).
             If None, no alpha balancing.
      gamma: focusing parameter. Common: 2.0.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = 0.25, eps: float = 1e-8):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.eps = eps
        self.bce = nn.BCEWithLogitsLoss(reduction="none")  # internally handles sigmoid

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        targets = targets.to(dtype=logits.dtype)

        # per-element BCE with logits
        bce = self.bce(logits, targets)

        # p_t = p if y=1 else (1-p)
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)

        focal_factor = (1.0 - p_t).clamp_min(0.0).pow(self.gamma)

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * focal_factor * bce
        else:
            loss = focal_factor * bce

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class BernoulliKLLoss(nn.Module):
    """
    Stable KL(p || q) for Bernoulli with soft targets p in [0,1] and q=sigmoid(logits).
    """

    def __init__(self, eps: float = 1e-6, clamp_logits: float | None = 20.0):
        super().__init__()
        self.eps = eps
        self.clamp_logits = clamp_logits

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.clamp_logits is not None:
            logits = logits.clamp(-self.clamp_logits, self.clamp_logits)

        # ensure float
        targets = targets.to(dtype=logits.dtype)

        # clamp targets to avoid log(0) / log(negative)
        p = targets.clamp(self.eps, 1.0 - self.eps)

        # stable log q and log(1-q)
        log_q = F.logsigmoid(logits)  # log(sigmoid(z))
        log_1mq = F.logsigmoid(-logits)  # log(1-sigmoid(z))

        # stable log p and log(1-p)
        log_p = torch.log(p)
        log_1mp = torch.log1p(-p)

        loss = p * (log_p - log_q) + (1.0 - p) * (log_1mp - log_1mq)

        if mask is None:
            return loss.mean()

        m = mask.to(dtype=loss.dtype)
        # TODO: test if really needed. Zero-out masked pixels safely (prevents NaNs in masked regions from propagating)
        loss = torch.where(m > 0, loss, torch.zeros_like(loss))

        denom = m.sum().clamp_min(self.eps)
        return loss.sum() / denom


class WeightedLoss(nn.Module):
    """
    Combine multiple loss modules with weights.

    All losses are expected to implement:
        forward(logits: Tensor, targets: Tensor, mask: Tensor|None) -> Tensor
    """

    losses: nn.ModuleDict
    _weights: torch.Tensor

    def __init__(
        self,
        losses: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
        normalize_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not losses:
            raise ValueError("losses must be a non-empty dict of name -> nn.Module")

        self.losses = nn.ModuleDict(losses)
        self.eps = eps
        self.normalize_weights = normalize_weights

        if weights is None:
            weights = {k: 1.0 for k in losses}

        missing = set(losses.keys()) - set(weights.keys())
        extra = set(weights.keys()) - set(losses.keys())
        if missing:
            raise ValueError(f"weights missing keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"weights has unknown keys: {sorted(extra)}")

        w = torch.tensor([weights[k] for k in losses], dtype=torch.float32)
        if self.normalize_weights:
            w = w / w.sum().clamp_min(self.eps)
        self.register_buffer("_weights", w)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self._weights

        total_loss = logits.new_tensor(0.0)
        loss_parts: dict[str, torch.Tensor] = {}

        for i, (name, loss_mod) in enumerate(self.losses.items()):
            loss_mod = cast(nn.Module, loss_mod)

            loss_val = cast(torch.Tensor, loss_mod(logits, targets, mask))
            loss_parts[name] = loss_val
            total_loss = total_loss + (w[i].to(dtype=loss_val.dtype) * loss_val)

        return total_loss, loss_parts
