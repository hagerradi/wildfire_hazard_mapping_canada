import os
import random

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from config import Config
from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.grid_loader.utils import get_range_burn_count, get_range_burn_prob
from data_preparation.utils import find_simulation_output_file
from logger import CometLogger
from losses import BCELoss, BernoulliKLLoss, DiceLoss, FocalLoss, MAELoss, MSELoss
from src.datasets.postprocessing.utils import get_predicted_hexel, save_predicted_hexels
from src.datasets.postprocessing.visualize_predictions import visualize_burn_prob_grids
from src.metrics import compute_bias, compute_mae, compute_mse, compute_spearman, compute_ssim

AVAILABLE_METRICS = {
    "mse": compute_mse,
    "mae": compute_mae,
    "spearman": compute_spearman,
    "ssim": compute_ssim,
    "bias": compute_bias,
}


def build_single_loss(name: str) -> torch.nn.Module:
    name = str(name).lower()
    if name in ["bce", "bceloss"]:
        return BCELoss()
    if name in ["mse", "mseloss"]:
        return MSELoss()
    if name in ["mae", "maeloss"]:
        return MAELoss()
    if name in ["focal", "focalloss"]:
        return FocalLoss()
    if name in ["dice", "diceloss"]:
        return DiceLoss()
    if name in ["klloss", "kl", "bernoullikl", "bernoulliklloss"]:
        return BernoulliKLLoss()
    raise ValueError(f"Unknown loss type: {name}")


def visualize_model_predictions(
    test_loader: DataLoader,
    test_predictions: np.ndarray,
    n_samples: int = 4,
    seed: int = 42,
    save_path: str = None,
    channel_map: dict = None,
    feature_names_list: list = None,
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
    seed: int, optional
        Random seed to get same patch IDs across different inference runs.
    save_path: str, optional
        Save path for the visualization figure (not saved if None).
    channel_map: dict, optional
        Dict mapping channel IDs to input feature names for plotting.
    feature_names_list: list, optional
        The list of used input features from the config. file.

    Returns
    -------
    None
        This function creates matplotlib figures and displays/saves them.
    """
    all_inputs, all_targets, all_masks = [], [], []

    # If more than one data source, we only need the grid for the viz.
    for batch in test_loader:
        if isinstance(batch, dict) and "grid" in batch:
            inputs, targets, masks = batch["grid"]
        else:
            inputs, targets, masks = batch

        all_inputs.append(inputs.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())
        all_masks.append(masks.detach().cpu().numpy())

    all_inputs = np.concatenate(all_inputs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_masks = np.concatenate(all_masks, axis=0)

    preds = test_predictions.squeeze(1) if test_predictions.ndim == 4 else test_predictions
    targets = all_targets.squeeze(1) if all_targets.ndim == 4 else all_targets
    masks = all_masks.squeeze(1) if all_masks.ndim == 4 else all_masks

    selected_indices = []
    idx_to_label = {}

    # use channel names map and config features list if we provide it
    if channel_map and feature_names_list:
        current_tensor_idx = 0
        for feat_name in feature_names_list:
            if feat_name not in channel_map:
                continue

            # see how many channels this feature originally had in the map
            orig_indices = channel_map[feat_name]
            num_channels_for_feat = len(orig_indices)

            # make new relative indices for the current data
            relative_indices = list(range(current_tensor_idx, current_tensor_idx + num_channels_for_feat))
            # for multichannel input feats, only show first and last as examples
            if num_channels_for_feat > 2:
                subset = [relative_indices[0], relative_indices[-1]]
                for i, rel_idx in enumerate(subset):
                    selected_indices.append(rel_idx)
                    suffix = "first" if i == 0 else "last"
                    idx_to_label[rel_idx] = f"{feat_name}\n({suffix})"
            else:
                for rel_idx in relative_indices:
                    selected_indices.append(rel_idx)
                    idx_to_label[rel_idx] = feat_name

            current_tensor_idx += num_channels_for_feat
    else:
        # if no map, we just print channel indices for the fig
        selected_indices = list(range(all_inputs.shape[1]))
        idx_to_label = {i: f"Ch {i}" for i in selected_indices}

    # check we aren't out of bounds after new mapping
    selected_indices = [idx for idx in selected_indices if idx < all_inputs.shape[1]]

    n_cols = len(selected_indices) + 2
    rng = np.random.RandomState(seed)  # fix seed to get same patch ids between inferences
    indices = rng.choice(preds.shape[0], n_samples, replace=False)

    _, axes = plt.subplots(n_samples, n_cols, figsize=(4.2 * n_cols, 4 * n_samples), dpi=300)
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for i, sample_idx in enumerate(indices):
        axes[i, 0].annotate(
            f"Patch ID: {sample_idx}",
            xy=(-0.5, 0.5),
            xycoords="axes fraction",
            ha="right",
            va="center",
            fontsize=14,
            fontweight="bold",
            rotation=90,
        )

        # show the input channels
        for col_idx, channel_idx in enumerate(selected_indices):
            ax = axes[i, col_idx]
            data = all_inputs[sample_idx, channel_idx]

            im = ax.imshow(data, cmap="viridis", vmin=data.min(), vmax=data.max())
            ax.set_title(idx_to_label[channel_idx], fontsize=10, fontweight="bold")
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        masked_pred = preds[sample_idx] * masks[sample_idx]
        target_data = targets[sample_idx]

        # Get min/max for consistent color scale for targets and preds
        v_min = min(masked_pred.min(), target_data.min())
        v_max = max(masked_pred.max(), target_data.max())

        # show targets
        ax_t = axes[i, n_cols - 2]
        im_t = ax_t.imshow(target_data, cmap="viridis", vmin=v_min, vmax=v_max)
        ax_t.set_title("Target", fontsize=10, fontweight="bold")
        ax_t.axis("off")
        plt.colorbar(im_t, ax=ax_t, fraction=0.046, pad=0.04)

        # show preds
        ax_p = axes[i, n_cols - 1]
        im_p = ax_p.imshow(masked_pred, cmap="viridis", vmin=v_min, vmax=v_max)
        ax_p.set_title("Prediction", fontsize=10, fontweight="bold")
        ax_p.axis("off")
        plt.colorbar(im_p, ax=ax_p, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=(0.05, 0, 1, 1))

    # save the viz fig for easier usage if set to True
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Visualization saved to: {save_path}")
        plt.close()
    else:
        plt.show()

    plt.tight_layout()
    plt.show()


def visualize_predicted_hexels(
    test_predictions: np.ndarray, config: Config, out_norm: str, experiment_logger: CometLogger | None = None
) -> None:
    """
    A util function to re-construct predicted hexels out of test predictions, and visualize side-by-side with the Groundtruth
    """
    data_dir = config.data.root_dir
    raw_data_dir = config.data.raw_data_dir
    modelling_approach = config.modelling_approach
    valid_mask_threshold = config.data.valid_mask_threshold
    output_type, season, cause = "prob", None, None
    if modelling_approach == "1":
        max_target_val, min_target_val = get_range_burn_prob(root_dir=raw_data_dir)
    else:
        max_target_val, min_target_val = get_range_burn_count(root_dir=raw_data_dir)

    if isinstance(test_predictions, str):
        # Handle the error or raise an exception
        raise TypeError(f"Expected ndarray, but got string: {test_predictions}")

    try:
        test_df = pd.read_csv(os.path.join(data_dir, config.data.test_split))
    except (FileNotFoundError, AttributeError):
        raise ValueError("Test df file does not exist.")  # noqa: B904

    # seperate hexels by their IDs
    test_df = test_df[test_df["valid_ratio"] > valid_mask_threshold].reset_index(drop=True)  # type: ignore
    all_hex_ids = list(test_df["hex_id"].unique())

    # loop over test hexels
    for hex_id in all_hex_ids:
        print(f"======Working with hex{hex_id}========")
        one_hexel_df = test_df[test_df["hex_id"] == hex_id]
        hexel_indices = test_df[test_df["hex_id"] == hex_id].index.tolist()
        if len(str(hex_id)) != 2:
            hex_id = "0" + str(hex_id)

        hex_test_predictions = test_predictions[hexel_indices]
        reconstructed_hexel_denorm, gt_elevation_grid_profile = get_predicted_hexel(
            base_dir=data_dir,
            raw_data_dir=raw_data_dir,
            test_df=one_hexel_df,
            predictions=hex_test_predictions,
            min_target_val=min_target_val,
            max_target_val=max_target_val,
            hex_id=hex_id,
            modelling_approach=modelling_approach,
            out_norm=out_norm,
            stitch_mode="mean",
            win_h=128,
            win_w=128,
        )
        save_predicted_hexels(reconstructed_hexel_denorm, gt_elevation_grid_profile, hex_id, config.save_dir)
        # Save the hex as plt plot
        hex_dir = os.path.join(raw_data_dir, f"hex{hex_id}")
        fpath = find_simulation_output_file(hex_dir, hex_id, output_type, season=season, cause=cause)
        grid_gt = load_output_burn_grid(fpath)

        visualize_burn_prob_grids(
            gt_grid=grid_gt,
            pred_grid=reconstructed_hexel_denorm,
            hex_id=hex_id,
            save_dir=config.save_dir,
            experiment_logger=experiment_logger,
        )

        print(f"=======Saved subplot for hex{hex_id}==============")


def seed_everything(seed: int = 42, deterministic: bool = True):
    """
    Seed all RNG sources for determinism.

    Parameters
    ----------
    seed: int
        Seed value
    determinstic: bool
        Ensures strict determinism but might slow down training

    Returns
    -------
    None
    """

    # (CPU) Python, OS, NumPy, Torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # (GPU, if available)
    if torch.cuda.is_available():
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi GPU in case
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    # MacOS / MPS specific
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    # For PyTorch >= 1.8
    # Outside 'if cuda' because PyTorch has deterministic CPU algorithms too.
    if deterministic:
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    print(f"[Info] Seed set to: {seed}")


def seed_worker(worker_id: int):
    """
    Helper function to set the seed for each worker based on the global seed.
    This ensures numpy and random in subprocesses are deterministic.
    """
    base_seed = torch.initial_seed()
    worker_seed = (base_seed + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def set_device() -> str:
    """Utils. to set up device."""
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = "mps"
    else:
        device = "cpu"

    return device
