import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader


def visualize_model_predictions(
    test_loader: DataLoader,
    test_predictions: np.ndarray,
    n_samples: int = 4,
) -> None:
    """
    Visualize model predictions versus targets for a selection of random samples.

    Parameters
    ----------
    test_loader : DataLoader
        DataLoader providing test batches as (inputs, targets, masks).
    test_predictions : numpy.ndarray
        Array containing model predictions corresponding to all samples in
        ``test_loader``, with shape ``(N, ...)`` or ``(N, 1, ...)``.
    n_samples : int, optional
        Number of random samples to visualize. Defaults to 4.

    Returns
    -------
    None
        This function creates matplotlib figures and displays them.
    """
    all_targets = []
    all_masks = []
    for batch in test_loader:
        _, targets, masks = batch  # (inputs, targets, mask)
        all_targets.append(targets.detach().numpy())
        all_masks.append(masks.detach().numpy())

    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    preds = test_predictions.squeeze(1) if test_predictions.ndim == 4 else test_predictions
    if all_targets.ndim == 4:
        all_targets = all_targets.squeeze(1)
    if all_masks.ndim == 4:
        all_masks = all_masks.squeeze(1)

    num_samples = preds.shape[0]
    indices = np.random.choice(num_samples, n_samples, replace=False)

    masked_preds = preds * all_masks
    fig, axes = plt.subplots(n_samples, 2, figsize=(15, 18))
    for i, idx in enumerate(indices):
        # Get min/max for consistent color scale
        vmin = min(masked_preds[idx].min(), all_targets[idx].min())
        vmax = max(masked_preds[idx].max(), all_targets[idx].max())

        # Plot target
        im0 = axes[i, 0].imshow(all_targets[idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"Target {idx}\nmin={all_targets[idx].min():.3f}, max={all_targets[idx].max():.3f}")
        axes[i, 0].axis("off")
        plt.colorbar(im0, ax=axes[i, 0], fraction=0.046, pad=0.04)

        # Plot masked prediction
        im1 = axes[i, 1].imshow(masked_preds[idx], cmap="viridis", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title(f"Prediction {idx} (masked)\nmin={masked_preds[idx].min():.3f}, max={masked_preds[idx].max():.3f}")
        axes[i, 1].axis("off")
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()
