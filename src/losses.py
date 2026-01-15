# Defintions of loss functions
import torch
import torch.nn as nn


class BCELoss(nn.Module):
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
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.mse(logits, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom


class MAELoss(nn.Module):
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mae = nn.L1Loss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.mae(logits, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)  # ensure float mask (1=valid, 0=invalid)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)

        return loss.sum() / denom
