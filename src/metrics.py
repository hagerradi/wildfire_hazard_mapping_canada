# Definitions of metrics for model evaluation

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.regression import spearman_corrcoef


def compute_mse(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    """Computes Mean Squared Error (MSE), optionally using a mask."""
    if mask is not None:
        preds = preds[~mask]
        targets = targets[~mask]
    return F.mse_loss(preds, targets)


def compute_mae(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    """Computes Mean Absolute Error (MAE), optionally using a mask."""
    if mask is not None:
        preds = preds[~mask]
        targets = targets[~mask]
    return F.l1_loss(preds, targets)


def compute_spearman(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    """
    Computes Spearman correlation per sample, then averages. Optionally uses a mask.
    """
    batch_size = preds.size(0)
    flat_preds = preds.view(batch_size, -1)
    flat_targets = targets.view(batch_size, -1)
    if mask is not None:
        flat_mask = mask.view(batch_size, -1)
        corrs = [spearman_corrcoef(flat_preds[i][~flat_mask[i]], flat_targets[i][~flat_mask[i]]) for i in range(batch_size)]
    else:
        corrs = [spearman_corrcoef(flat_preds[i], flat_targets[i]) for i in range(batch_size)]
    return torch.stack(corrs).mean()


def compute_ssim(preds: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor = None):
    """
    Computes Structural Similarity Index Measure (SSIM), optionally using a mask.
    """
    # we assume here our preds range will be between 0-1
    if mask is not None:
        preds = preds.clone()
        targets = targets.clone()
        preds[mask] = 0.0
        targets[mask] = 0.0
    return structural_similarity_index_measure(preds, targets, data_range=1.0)
