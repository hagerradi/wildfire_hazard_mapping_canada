# Definitions of metrics for model evaluation

import torch
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torchmetrics.functional.regression import spearman_corrcoef

def compute_mse(preds: torch.Tensor, targets: torch.Tensor):
    """Mean Squared Error"""
    return F.mse_loss(preds, targets)

def compute_mae(preds: torch.Tensor, targets: torch.Tensor):
    """Mean Absolute Error"""
    return F.l1_loss(preds, targets)

def compute_spearman(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates Spearman correlation per sample, then averages.
    """
    # Generate flattened preds and targets
    batch_size = preds.size(0)
    flat_preds = preds.view(batch_size, -1)
    flat_targets = targets.view(batch_size, -1)
    
    # We compute spearman between each pair of preds-targets, then average
    corrs = [spearman_corrcoef(flat_preds[i], flat_targets[i]) for i in range(batch_size)]
    return torch.tensor(corrs, device=preds.device).mean()

def compute_ssim(preds: torch.Tensor, targets: torch.Tensor):
    """
    Calculates SSIM.
    """
    return structural_similarity_index_measure(preds, targets, data_range=1.0)

if __name__ == "__main__":
    
    target = torch.rand(8, 1, 64, 64)
    pred = target * 0.75

    print(f"MSE: {compute_mse(pred, target):.4f}")
    print(f"MAE: {compute_mae(pred, target):.4f}")
    print(f"Spearman: {compute_spearman(pred, target):.4f}")
    print(f"SSIM: {compute_ssim(pred, target):.4f}")