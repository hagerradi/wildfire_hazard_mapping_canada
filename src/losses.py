# Definitions of loss functions
from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(loss: torch.Tensor, mask: torch.Tensor | None, eps: float) -> torch.Tensor:
    if mask is None:
        return loss.mean()
    mask = mask.to(dtype=loss.dtype)
    return (loss * mask).sum() / mask.sum().clamp_min(eps)


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


class RegressionMSELoss(nn.Module):
    """
    MSE loss on raw model outputs with optional mask.

    This is intended for unconstrained regression targets, e.g. standardized
    log FI/ROS. Use MSELoss for probability targets that need sigmoid outputs.
    """

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.mse = nn.MSELoss(reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.mse(logits, targets)

        if mask is None:
            return loss.mean()

        mask = mask.to(dtype=loss.dtype)
        loss = loss * mask
        denom = mask.sum().clamp_min(self.eps)
        return loss.sum() / denom


class CCCLoss(nn.Module):
    """
    Concordance correlation coefficient loss, 1 - CCC.

    Use the sigmoid variant for probability targets such as BP. Use
    RegressionCCCLoss for raw regression outputs such as log-standard FI/ROS.
    """

    def __init__(self, use_sigmoid: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        preds = preds.flatten(1).float()
        targets = targets.flatten(1).float()
        mask_flat = None if mask is None else mask.bool().flatten(1)

        ccc_values = []
        for idx in range(preds.shape[0]):
            pred_i = preds[idx]
            target_i = targets[idx]
            if mask_flat is not None:
                valid = mask_flat[idx]
                pred_i = pred_i[valid]
                target_i = target_i[valid]
            if pred_i.numel() < 2:
                continue
            pred_mean = pred_i.mean()
            target_mean = target_i.mean()
            covariance = ((pred_i - pred_mean) * (target_i - target_mean)).mean()
            denominator = pred_i.var(correction=0) + target_i.var(correction=0) + (pred_mean - target_mean).pow(2)
            ccc_values.append(2.0 * covariance / denominator.clamp_min(self.eps))

        if not ccc_values:
            return logits.new_tensor(0.0)
        return 1.0 - torch.stack(ccc_values).mean()


class RegressionCCCLoss(CCCLoss):
    """CCC loss on raw model outputs."""

    def __init__(self, eps: float = 1e-8):
        super().__init__(use_sigmoid=False, eps=eps)


class PearsonLoss(nn.Module):
    """
    Pearson correlation loss, 1 - r.

    This is a differentiable rank-order proxy. Use the sigmoid variant for BP
    and RegressionPearsonLoss for raw FI/ROS regression outputs.
    """

    def __init__(self, use_sigmoid: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_sigmoid = use_sigmoid
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        preds = preds.flatten(1).float()
        targets = targets.flatten(1).float()
        mask_flat = None if mask is None else mask.bool().flatten(1)

        correlations = []
        for idx in range(preds.shape[0]):
            pred_i = preds[idx]
            target_i = targets[idx]
            if mask_flat is not None:
                valid = mask_flat[idx]
                pred_i = pred_i[valid]
                target_i = target_i[valid]
            if pred_i.numel() < 2:
                continue
            pred_centered = pred_i - pred_i.mean()
            target_centered = target_i - target_i.mean()
            denom = pred_centered.norm() * target_centered.norm()
            correlations.append((pred_centered * target_centered).sum() / denom.clamp_min(self.eps))

        if not correlations:
            return logits.new_tensor(0.0)
        return 1.0 - torch.stack(correlations).mean()


class RegressionPearsonLoss(PearsonLoss):
    """Pearson loss on raw model outputs."""

    def __init__(self, eps: float = 1e-8):
        super().__init__(use_sigmoid=False, eps=eps)


class LogCoshLoss(nn.Module):
    """Stable log-cosh loss on raw model outputs with optional mask."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        error = logits - targets
        loss = error + F.softplus(-2.0 * error) - torch.log(error.new_tensor(2.0))
        return _masked_mean(loss, mask, self.eps)


class TailWeightedHuberLoss(nn.Module):
    """Huber loss with target-tail pixels up-weighted by a batch percentile."""

    def __init__(self, beta: float = 1.0, percentile: float = 0.75, tail_weight: float = 5.0, eps: float = 1e-8):
        super().__init__()
        if beta <= 0.0:
            raise ValueError(f"Huber beta must be positive, got {beta}.")
        if not 0.0 < percentile < 1.0:
            raise ValueError(f"Tail percentile must be between 0 and 1, got {percentile}.")
        if tail_weight < 1.0:
            raise ValueError(f"Tail weight must be at least 1.0, got {tail_weight}.")
        self.beta = beta
        self.percentile = percentile
        self.tail_weight = tail_weight
        self.eps = eps
        self.huber = nn.SmoothL1Loss(beta=beta, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        loss = self.huber(logits, targets)
        valid_targets = targets.flatten() if mask is None else targets[mask.bool()]
        if valid_targets.numel() == 0:
            return loss.sum() * 0.0
        threshold = torch.quantile(valid_targets.float(), self.percentile)
        weights = 1.0 + (self.tail_weight - 1.0) * (targets >= threshold).to(dtype=loss.dtype)
        return _masked_mean(loss * weights, mask, self.eps)


class QuantileLoss(nn.Module):
    """Pinball loss on raw model outputs."""

    def __init__(self, quantile: float = 0.85, eps: float = 1e-8):
        super().__init__()
        if not 0.0 < quantile < 1.0:
            raise ValueError(f"Quantile must be between 0 and 1, got {quantile}.")
        self.quantile = quantile
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
        diff = targets - logits
        loss = torch.maximum(self.quantile * diff, (self.quantile - 1.0) * diff)
        return _masked_mean(loss, mask, self.eps)


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
        if beta <= 0.0:
            raise ValueError(f"Huber beta must be positive, got {beta}.")
        self.eps = eps
        self.beta = beta
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


class HexSummaryLoss(nn.Module):
    """Differentiable cross-hex loss on batch-level BP summaries.

    This is not a stitched-raster loss. It uses patch metadata to group patches
    by hex ID within each batch, summarizes valid BP predictions per patch, then
    compares grouped hex summaries. It is intended as a lightweight proxy for
    the stitched/hex ranking failure mode.
    """

    requires_patch_metadata = True

    def __init__(
        self,
        summary: str = "mean",
        correlation: str = "pearson",
        top_fraction: float = 0.10,
        rank_temperature: float = 1.0,
        min_target_gap: float = 1e-6,
        rank_scale_min: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.summary = summary.lower()
        self.correlation = correlation.lower()
        self.top_fraction = top_fraction
        self.rank_temperature = rank_temperature
        self.min_target_gap = min_target_gap
        self.rank_scale_min = rank_scale_min
        self.eps = eps

        if self.summary not in {"mean", "topk_mean"}:
            raise ValueError(f"Unsupported hex summary={summary!r}. Use 'mean' or 'topk_mean'.")
        if self.correlation not in {"pearson", "ccc", "pairwise_rank"}:
            raise ValueError(f"Unsupported hex summary correlation={correlation!r}. Use 'pearson', 'ccc', or 'pairwise_rank'.")
        if not 0.0 < self.top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
        if self.rank_temperature <= 0.0:
            raise ValueError(f"rank_temperature must be positive, got {rank_temperature}.")
        if self.min_target_gap < 0.0:
            raise ValueError(f"min_target_gap must be non-negative, got {min_target_gap}.")
        if self.rank_scale_min <= 0.0:
            raise ValueError(f"rank_scale_min must be positive, got {rank_scale_min}.")

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if logits.shape[1] != 1:
            raise ValueError(f"HexSummaryLoss supports single-target BP only, got logits shape {tuple(logits.shape)}.")
        if patch_metadata is None or "hex_id" not in patch_metadata:
            raise ValueError("HexSummaryLoss requires patch_metadata containing a 'hex_id' tensor.")

        hex_ids = patch_metadata["hex_id"].to(device=logits.device)
        if hex_ids.ndim != 1 or hex_ids.shape[0] != logits.shape[0]:
            raise ValueError(f"Expected hex_id shape ({logits.shape[0]},), got {tuple(hex_ids.shape)}.")

        probs = torch.sigmoid(logits)
        patch_pred, patch_target = self._patch_summaries(probs, targets, mask)
        if patch_pred.numel() < 2:
            return logits.new_tensor(0.0)

        hex_pred = []
        hex_target = []
        for hex_id in torch.unique(hex_ids):
            group = hex_ids == hex_id
            if not group.any():
                continue
            hex_pred.append(patch_pred[group].mean())
            hex_target.append(patch_target[group].mean())

        if len(hex_pred) < 2:
            return logits.new_tensor(0.0)

        pred = torch.stack(hex_pred).float()
        target = torch.stack(hex_target).float()
        if self.correlation == "pearson":
            return self._pearson_loss(pred, target)
        if self.correlation == "ccc":
            return self._ccc_loss(pred, target)
        return self._pairwise_rank_loss(pred, target)

    def _patch_summaries(
        self,
        probs: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pred_values = []
        target_values = []
        mask_bool = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()

        for idx in range(probs.shape[0]):
            valid = mask_bool[idx, 0]
            pred_i = probs[idx, 0][valid]
            target_i = targets[idx, 0][valid]
            if pred_i.numel() == 0:
                pred_values.append(probs.new_tensor(0.0))
                target_values.append(targets.new_tensor(0.0))
                continue
            if self.summary == "mean":
                pred_values.append(pred_i.mean())
                target_values.append(target_i.mean())
                continue

            k = max(1, int(torch.ceil(target_i.new_tensor(float(target_i.numel() * self.top_fraction))).item()))
            top_indices = torch.topk(target_i, k=k, largest=True).indices
            pred_values.append(pred_i[top_indices].mean())
            target_values.append(target_i[top_indices].mean())

        return torch.stack(pred_values), torch.stack(target_values)

    def _pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_centered = pred - pred.mean()
        target_centered = target - target.mean()
        denom = pred_centered.norm() * target_centered.norm()
        return 1.0 - (pred_centered * target_centered).sum() / denom.clamp_min(self.eps)

    def _ccc_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_mean = pred.mean()
        target_mean = target.mean()
        covariance = ((pred - pred_mean) * (target - target_mean)).mean()
        denominator = pred.var(correction=0) + target.var(correction=0) + (pred_mean - target_mean).pow(2)
        return 1.0 - 2.0 * covariance / denominator.clamp_min(self.eps)

    def _pairwise_rank_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_diff = pred[:, None] - pred[None, :]
        target_diff = target[:, None] - target[None, :]
        upper_triangular = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        valid = upper_triangular & (target_diff.abs() > self.min_target_gap)
        if not valid.any():
            return pred.new_tensor(0.0)

        direction = target_diff[valid].sign()
        pred_scale = pred.std(correction=0).detach().clamp_min(self.rank_scale_min)
        ordered_margin = direction * (pred_diff[valid] / pred_scale) / self.rank_temperature
        return F.softplus(-ordered_margin).mean()


class SampledCellRankLoss(nn.Module):
    """Pairwise rank loss on sampled valid cells, focused on target hotspots.

    The loss samples top-target cells plus a small reference set from each hex in
    the batch, then penalizes prediction order inversions. It is intentionally
    sampled: full cell-pair ranking is quadratic in valid pixels and too costly.
    """

    requires_patch_metadata = True

    def __init__(
        self,
        use_sigmoid: bool = True,
        top_fraction: float = 0.10,
        top_samples_per_group: int = 64,
        reference_samples_per_group: int = 64,
        max_pairs_per_group: int = 4096,
        min_target_gap: float = 1e-6,
        rank_temperature: float = 1.0,
        rank_scale_min: float = 1e-3,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not 0.0 < top_fraction <= 1.0:
            raise ValueError(f"top_fraction must be in (0, 1], got {top_fraction}.")
        if top_samples_per_group < 1:
            raise ValueError(f"top_samples_per_group must be positive, got {top_samples_per_group}.")
        if reference_samples_per_group < 1:
            raise ValueError(f"reference_samples_per_group must be positive, got {reference_samples_per_group}.")
        if max_pairs_per_group < 1:
            raise ValueError(f"max_pairs_per_group must be positive, got {max_pairs_per_group}.")
        if min_target_gap < 0.0:
            raise ValueError(f"min_target_gap must be non-negative, got {min_target_gap}.")
        if rank_temperature <= 0.0:
            raise ValueError(f"rank_temperature must be positive, got {rank_temperature}.")
        if rank_scale_min <= 0.0:
            raise ValueError(f"rank_scale_min must be positive, got {rank_scale_min}.")

        self.use_sigmoid = use_sigmoid
        self.top_fraction = top_fraction
        self.top_samples_per_group = top_samples_per_group
        self.reference_samples_per_group = reference_samples_per_group
        self.max_pairs_per_group = max_pairs_per_group
        self.min_target_gap = min_target_gap
        self.rank_temperature = rank_temperature
        self.rank_scale_min = rank_scale_min
        self.eps = eps

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if logits.shape[1] != 1:
            raise ValueError(f"SampledCellRankLoss supports single-target outputs only, got logits shape {tuple(logits.shape)}.")
        if patch_metadata is None or "hex_id" not in patch_metadata:
            raise ValueError("SampledCellRankLoss requires patch_metadata containing a 'hex_id' tensor.")

        hex_ids = patch_metadata["hex_id"].to(device=logits.device)
        if hex_ids.ndim != 1 or hex_ids.shape[0] != logits.shape[0]:
            raise ValueError(f"Expected hex_id shape ({logits.shape[0]},), got {tuple(hex_ids.shape)}.")

        preds = torch.sigmoid(logits) if self.use_sigmoid else logits
        mask_bool = torch.ones_like(targets, dtype=torch.bool) if mask is None else mask.bool()
        losses = []

        for hex_id in torch.unique(hex_ids):
            group = hex_ids == hex_id
            pred_values = preds[group, 0][mask_bool[group, 0]].float()
            target_values = targets[group, 0][mask_bool[group, 0]].float()
            group_loss = self._group_loss(pred_values, target_values)
            if group_loss is not None:
                losses.append(group_loss)

        if not losses:
            return logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _group_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor | None:
        n_values = target.numel()
        if n_values < 2:
            return None

        top_count = min(self.top_samples_per_group, max(1, int(torch.ceil(target.new_tensor(n_values * self.top_fraction)).item())))
        top_indices = torch.topk(target, k=top_count, largest=True).indices

        top_mask = torch.zeros(n_values, dtype=torch.bool, device=target.device)
        top_mask[top_indices] = True
        reference_pool = (~top_mask).nonzero(as_tuple=False).flatten()
        if reference_pool.numel() == 0:
            candidate_indices = top_indices
            candidate_is_top = torch.ones_like(candidate_indices, dtype=torch.bool)
        else:
            reference_count = min(self.reference_samples_per_group, reference_pool.numel())
            if reference_count == reference_pool.numel():
                reference_indices = reference_pool
            else:
                positions = torch.linspace(
                    0,
                    reference_pool.numel() - 1,
                    steps=reference_count,
                    device=target.device,
                ).round()
                reference_indices = reference_pool[positions.long()]
            candidate_indices = torch.cat([top_indices, reference_indices])
            candidate_is_top = torch.cat(
                [
                    torch.ones(top_indices.numel(), dtype=torch.bool, device=target.device),
                    torch.zeros(reference_indices.numel(), dtype=torch.bool, device=target.device),
                ]
            )

        if candidate_indices.numel() < 2:
            return None

        pred_candidate = pred[candidate_indices]
        target_candidate = target[candidate_indices]
        pred_diff = pred_candidate[:, None] - pred_candidate[None, :]
        target_diff = target_candidate[:, None] - target_candidate[None, :]

        upper_triangular = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        includes_hotspot = candidate_is_top[:, None] | candidate_is_top[None, :]
        valid = upper_triangular & includes_hotspot & (target_diff.abs() > self.min_target_gap)
        if not valid.any():
            return None

        pred_pairs = pred_diff[valid]
        target_pairs = target_diff[valid]
        if pred_pairs.numel() > self.max_pairs_per_group:
            positions = torch.linspace(
                0,
                pred_pairs.numel() - 1,
                steps=self.max_pairs_per_group,
                device=pred_pairs.device,
            ).round()
            selected = positions.long()
            pred_pairs = pred_pairs[selected]
            target_pairs = target_pairs[selected]

        direction = target_pairs.sign()
        pred_scale = pred_candidate.std(correction=0).detach().clamp_min(self.rank_scale_min)
        ordered_margin = direction * (pred_pairs / pred_scale) / self.rank_temperature
        return F.softplus(-ordered_margin).mean()


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
        self.requires_patch_metadata = any(getattr(loss_mod, "requires_patch_metadata", False) for loss_mod in self.losses.values())

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
        patch_metadata: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        w = self._weights

        total_loss = logits.new_tensor(0.0)
        loss_parts: dict[str, torch.Tensor] = {}

        for i, (name, loss_mod) in enumerate(self.losses.items()):
            loss_mod = cast(nn.Module, loss_mod)

            if getattr(loss_mod, "requires_patch_metadata", False):
                loss_val = cast(torch.Tensor, loss_mod(logits, targets, mask, patch_metadata=patch_metadata))
            else:
                loss_val = cast(torch.Tensor, loss_mod(logits, targets, mask))
            loss_parts[name] = loss_val
            total_loss = total_loss + (w[i].to(dtype=loss_val.dtype) * loss_val)

        return total_loss, loss_parts


class MultiTargetLoss(nn.Module):
    """Combines target-specific losses for multi-output BP/FI/ROS models."""

    losses: nn.ModuleDict
    _weights: torch.Tensor

    def __init__(
        self,
        target_names: list[str],
        losses: dict[str, nn.Module],
        weights: dict[str, float] | None = None,
        normalize_weights: bool = True,
        eps: float = 1e-8,
    ):
        super().__init__()
        if not target_names:
            raise ValueError("target_names must be non-empty.")
        if set(target_names) != set(losses):
            raise ValueError(f"losses must match target_names. target_names={target_names}, losses={sorted(losses)}")

        self.target_names = list(target_names)
        self.losses = nn.ModuleDict({name: losses[name] for name in self.target_names})
        self.eps = eps
        self.normalize_weights = normalize_weights

        if weights is None:
            weights = {name: 1.0 for name in self.target_names}

        missing = set(self.target_names) - set(weights)
        extra = set(weights) - set(self.target_names)
        if missing:
            raise ValueError(f"weights missing target keys: {sorted(missing)}")
        if extra:
            raise ValueError(f"weights has unknown target keys: {sorted(extra)}")

        weight_values = torch.tensor([weights[name] for name in self.target_names], dtype=torch.float32)
        if self.normalize_weights:
            weight_values = weight_values / weight_values.sum().clamp_min(self.eps)
        self.register_buffer("_weights", weight_values)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if logits.shape[1] != len(self.target_names):
            raise ValueError(f"Expected {len(self.target_names)} output channels, got logits shape {tuple(logits.shape)}.")
        if targets.shape[1] != len(self.target_names):
            raise ValueError(f"Expected {len(self.target_names)} target channels, got targets shape {tuple(targets.shape)}.")
        if mask is not None and mask.shape[1] != len(self.target_names):
            raise ValueError(f"Expected {len(self.target_names)} mask channels, got mask shape {tuple(mask.shape)}.")

        total_loss = logits.new_tensor(0.0)
        loss_parts: dict[str, torch.Tensor] = {}
        for idx, name in enumerate(self.target_names):
            channel_mask = None if mask is None else mask[:, idx : idx + 1]
            loss_val = cast(
                torch.Tensor,
                self.losses[name](logits[:, idx : idx + 1], targets[:, idx : idx + 1], channel_mask),
            )
            loss_parts[name] = loss_val
            total_loss = total_loss + (self._weights[idx].to(dtype=loss_val.dtype) * loss_val)

        return total_loss, loss_parts
