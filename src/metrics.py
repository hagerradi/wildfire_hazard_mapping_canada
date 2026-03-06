# Definitions of metrics for model evaluation
import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.regression import spearman_corrcoef


def compute_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """Computes Mean Squared Error (MSE), optionally using a mask."""
    if mask is None:
        return F.mse_loss(preds, targets, reduction="mean")

    mask = mask.to(dtype=preds.dtype)
    loss = F.mse_loss(preds, targets, reduction="none")

    # apply mask + normalize by valid count
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def compute_mae(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None, eps: float = 1e-8) -> torch.Tensor:
    """Computes Mean Absolute Error (MAE), optionally using a mask."""
    if mask is None:
        return F.l1_loss(preds, targets, reduction="mean")

    mask = mask.to(dtype=preds.dtype)
    loss = F.l1_loss(preds, targets, reduction="none")

    # apply mask + normalize by valid count
    loss = loss * mask
    denom = mask.sum().clamp_min(eps)
    return loss.sum() / denom


def compute_spearman(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
    """
    Computes Spearman correlation per sample, then averages. Optionally uses a mask.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    min_valid = 2

    corrs = []
    if mask is None:
        for i in range(batch_size):
            corrs.append(spearman_corrcoef(flat_preds[i], flat_targets[i]))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)  # True = valid
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]
            if sample_valid_mask.sum() < min_valid:
                corrs.append(torch.tensor(float("nan"), device=preds.device))
                continue
            corrs.append(spearman_corrcoef(flat_preds[i][sample_valid_mask], flat_targets[i][sample_valid_mask]))

    return torch.nanmean(torch.stack(corrs))


def compute_ssim(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    Computes SSIM over valid pixels only.
    Assumes preds/targets ∈ [0, 1].
    """
    if mask is None:
        return structural_similarity_index_measure(preds, targets, data_range=1.0)  # type: ignore

    # Ensure mask is boolean and broadcastable
    mask_bool = mask.bool()
    # If mask is missing channel dim, unsqueeze to match preds/targets
    while mask_bool.dim() < preds.dim():
        mask_bool = mask_bool.unsqueeze(1)

    # If all masked, return nan
    if mask_bool.sum() == 0:
        return torch.tensor(float("nan"), device=preds.device)

    # Set masked (invalid) pixels to 0 (or another constant)
    preds_masked = preds.clone().masked_fill(~mask_bool, 0.0)
    targets_masked = targets.clone().masked_fill(~mask_bool, 0.0)

    return structural_similarity_index_measure(preds_masked, targets_masked, data_range=1.0)  # type: ignore


def compute_bias(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    preds, targets, mask: same shape
    returns scalar bias (mean(preds-target) over valid pixels)
    """
    preds = preds.float()
    targets = targets.float()
    m = (mask > 0).float()

    denom = m.sum().clamp_min(1.0)  # avoid divide-by-zero
    return ((preds - targets) * m).sum() / denom


def compute_topK_iou(
    preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None, percentile: float = 0.90, eps: float = 1e-8
) -> torch.Tensor:
    """
    Computes the Intersection over Union (IoU) on binarized top K percentile maps.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    ious = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]

            p_thresh = torch.quantile(p.float(), percentile)
            t_thresh = torch.quantile(t.float(), percentile)

            # binarization
            p_bin = p >= p_thresh
            t_bin = t >= t_thresh

            intersection = (p_bin & t_bin).sum().float()
            union = (p_bin | t_bin).sum().float()

            ious.append(intersection / (union + eps))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]

            if sample_valid_mask.sum() == 0:
                ious.append(torch.tensor(float("nan"), device=preds.device))
                continue

            p_valid = flat_preds[i][sample_valid_mask]
            t_valid = flat_targets[i][sample_valid_mask]

            p_thresh = torch.quantile(p_valid.float(), percentile)
            t_thresh = torch.quantile(t_valid.float(), percentile)

            p_bin = p_valid >= p_thresh
            t_bin = t_valid >= t_thresh

            intersection = (p_bin & t_bin).sum().float()
            union = (p_bin | t_bin).sum().float()

            ious.append(intersection / (union + eps))

    return torch.nanmean(torch.stack(ious))


def compute_auc_iou(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor = None,
    k_values: tuple[float, float] = (0.01, 0.99),
    steps: int = 99,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Computes the Area Under the Curve (AUC) for IoU for a specified continuous TopK perc. range.

    Args:
        k_values (tuple[float, float]): Tuple (min_k, max_k) defining the continuous range for eval.
        steps (int): Number of points to evaluate within the continuous range.
    """
    batch_size = preds.size(0)
    flat_preds = preds.reshape(batch_size, -1)
    flat_targets = targets.reshape(batch_size, -1)

    # validations of inputs
    if not (isinstance(k_values, tuple) and len(k_values) == 2):
        raise ValueError("k_values must be a tuple of (min_k, max_k).")
    if not all(isinstance(k, (int, float)) for k in k_values):
        raise ValueError("k_values must contain numeric values (int or float).")
    min_k, max_k = float(k_values[0]), float(k_values[1])
    if not (0.0 < min_k <= 1.0 and 0.0 < max_k <= 1.0):
        raise ValueError("Each value in k_values must be within the open-closed interval (0, 1].")
    if not min_k < max_k:
        raise ValueError("k_values must satisfy min_k < max_k.")
    if not isinstance(steps, int) or steps < 2:
        raise ValueError("steps must be an integer greater than or equal to 2.")

    k_tensor = torch.linspace(min_k, max_k, steps=steps, device=preds.device)
    percentiles = 1.0 - k_tensor

    aucs = []

    if mask is None:
        for i in range(batch_size):
            p = flat_preds[i]
            t = flat_targets[i]

            p_thresh = torch.quantile(p.float(), percentiles)
            t_thresh = torch.quantile(t.float(), percentiles)

            p_bin = p.unsqueeze(0) >= p_thresh.unsqueeze(1)
            t_bin = t.unsqueeze(0) >= t_thresh.unsqueeze(1)

            intersection = (p_bin & t_bin).sum(dim=1).float()
            union = (p_bin | t_bin).sum(dim=1).float()

            ious = intersection / (union + eps)

            auc = torch.trapz(ious, k_tensor)
            max_area = k_tensor[-1] - k_tensor[0]

            aucs.append(auc / max_area if max_area > 0 else torch.tensor(float("nan"), device=preds.device))
    else:
        valid_mask = mask.bool().reshape(batch_size, -1)
        for i in range(batch_size):
            sample_valid_mask = valid_mask[i]

            if sample_valid_mask.sum() == 0:
                aucs.append(torch.tensor(float("nan"), device=preds.device))
                continue

            p_valid = flat_preds[i][sample_valid_mask]
            t_valid = flat_targets[i][sample_valid_mask]

            p_thresh = torch.quantile(p_valid.float(), percentiles)
            t_thresh = torch.quantile(t_valid.float(), percentiles)

            p_bin = p_valid.unsqueeze(0) >= p_thresh.unsqueeze(1)
            t_bin = t_valid.unsqueeze(0) >= t_thresh.unsqueeze(1)

            intersection = (p_bin & t_bin).sum(dim=1).float()
            union = (p_bin | t_bin).sum(dim=1).float()

            ious = intersection / (union + eps)

            auc = torch.trapz(ious, k_tensor)
            max_area = k_tensor[-1] - k_tensor[0]

            aucs.append(auc / max_area if max_area > 0 else torch.tensor(float("nan"), device=preds.device))

    return torch.nanmean(torch.stack(aucs))
