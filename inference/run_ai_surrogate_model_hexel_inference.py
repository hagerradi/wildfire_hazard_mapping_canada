"""
End-to-end inference pipeline for a single hexel.
Orchestrates data preparation, dataset building, and prediction.
"""

import argparse
import logging
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_preparation.grid_loader.output import load_output_burn_grid
from data_preparation.hexel_loader import load_features_per_hexel
from data_preparation.process_hexels_into_grids import get_split_hexel_window
from data_preparation.process_tabular_data import build_weather_table, process_fire_size_distribution_table
from data_preparation.utils import find_hex_ids, find_simulation_output_file
from inference.predictor import BurnRiskPredictor
from src.datasets.dataset import MultiSourceDataset
from src.datasets.postprocessing.utils import get_predicted_hexel, save_predicted_hexels, visualize_burn_prob_grids
from src.datasets.utils import get_data_source_class, get_data_source_param_class, get_dataset_dimensions

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("inference/run_hexel_inference.log", mode="w"),
    ],
    force=True,
)
logger = logging.getLogger(__name__)


def prepare_hexel_data(
    data_dir: Path,
    hex_id: str,
    win_h: int = 128,
    win_w: int = 128,
    overlap_ratio: float = 0.2,
    modelling_approach: int = 1,
    output_type: str = "prob",
    weather_sampling: str = "weather_zone_id",
) -> Path:
    """
    Prepare data patches for a single hexel.

    Handles all the CSV building, raw data loading, and patch splitting.

    Args:
        data_dir: Directory containing hexel data.
        hex_id: Hexel ID to process (e.g., "02").
        win_h: Patch height in pixels.
        win_w: Patch width in pixels.
        overlap_ratio: Overlap between patches (0.0 to 1.0).
        modelling_approach: 1 for joint season-cause, 2 for separate.
        output_type: "count" or "prob" for fire output type.
        weather_sampling: Weather sampling strategy.

    Returns:
        Path to the output directory containing patches and metadata CSV.
    """
    processed_data_dir = data_dir / f"data_samples_approach_{modelling_approach}"
    processed_data_dir.mkdir(parents=True, exist_ok=True)
    (processed_data_dir / "numpy_files").mkdir(parents=True, exist_ok=True)

    weather_table_path = processed_data_dir / "weather_table_processed.csv"
    logger.info("Building weather table...")
    build_weather_table(root_dir=data_dir, save_path=weather_table_path)

    fire_size_input = data_dir / "df_fire_fru.csv"
    fire_size_output = processed_data_dir / "df_fire_fru_processed.csv"
    if fire_size_input.exists():
        logger.info("Processing fire size distribution table...")
        process_fire_size_distribution_table(input_path=fire_size_input, output_path=fire_size_output)

    # Load features for the hexel
    feature_channel_map_path = processed_data_dir / f"feature_channel_map_{modelling_approach}.json"

    available_hex_ids = find_hex_ids(str(data_dir))
    if hex_id not in available_hex_ids:
        logger.error(f"Hexel ID {hex_id} not found in {data_dir}. Available hexel IDs: {available_hex_ids}")
        raise ValueError(f"Hexel ID {hex_id} not found in {data_dir}. Check logs for details.")

    logger.info(f"Loading features for hexel {hex_id}...")
    stacked_feats, mask, season_cause_mapping = load_features_per_hexel(
        root_dir=str(data_dir),
        hex_id=hex_id,
        feature_channel_map_path=str(feature_channel_map_path),
        modelling_approach=modelling_approach,
        output_type=output_type,
        weather_sampling=weather_sampling,
    )

    if stacked_feats is None or mask is None:
        logger.error(f"Failed to load features or mask for hexel {hex_id}. Aborting data preparation.")
        raise ValueError(f"Failed to load features or mask for hexel {hex_id}. Check logs for details.")

    logger.info(f"Splitting hexel {hex_id} into {win_h}x{win_w} patches...")
    get_split_hexel_window(
        season_cause_stacked_feats=stacked_feats,
        season_cause_mask=mask,
        season_cause_mapping=season_cause_mapping,
        out_dir=str(processed_data_dir),
        root_dir=str(data_dir),
        hex_id=hex_id,
        win_h=win_h,
        win_w=win_w,
        overlap_ratio=overlap_ratio,
    )

    logger.info(f"Data preparation complete. Output saved to {processed_data_dir}")
    return processed_data_dir


def create_dataset(processed_data_dir: Path, hex_id: str, config_dict: dict) -> MultiSourceDataset:
    """
    Build the PyTorch Dataset based on the saved checkpoint config.

    Args:
        processed_data_dir: Directory containing processed data.
        hex_id: Hexel ID.
        config_dict: Checkpoint config dict.

    Returns:
        MultiSourceDataset ready for inference.
    """
    csv_name = f"meta_hex_{hex_id}.csv"
    filename_col = config_dict["filename_col"]
    valid_mask_threshold = config_dict["valid_mask_threshold"]
    sources = {}
    for source in config_dict["input_sources"]:
        source_name = source["name"]
        source_class = get_data_source_class(source_name)
        source_param_class = get_data_source_param_class(source_name)
        sources[source_name] = source_class(root_dir=processed_data_dir, params=source_param_class(**source["params"]))

    return MultiSourceDataset(
        csv_name=csv_name,
        root_dir=str(processed_data_dir),
        sources=sources,
        filename_col=filename_col,
        valid_mask_threshold=valid_mask_threshold,
    )


def run_single_hexel_pipeline(
    checkpoint_path: Path,
    data_dir: Path,
    hex_id: str,
    batch_size: int = 32,
    num_workers: int = 4,
    prepare_data: bool = False,
    save_dir: Path = Path("outputs"),
) -> tuple[np.ndarray, Any]:
    """
    Orchestrate the end-to-end (data preparation + inference + post-processing) for one specific hexel.

    Args:
        checkpoint_path: Path to trained model checkpoint.
        data_dir: Directory containing hexel data.
        hex_id: Hexel ID to process.
        batch_size: Batch size for inference.
        num_workers: Number of dataloader workers.
        prepare_data: If True, run data preparation step.
        win_h: Patch height in pixels.
        win_w: Patch width in pixels.
        overlap_ratio: Overlap ratio between patches.
        modelling_approach: 1 for joint season-cause, 2 for separate.
        output_type: "count" or "prob" for fire output.
        weather_sampling: Weather sampling strategy.
        save_dir: Directory to save predictions and visualizations.

    Returns:
        Reconstructed hexel grid of burn probabilities (denormalized), and the ground truth elevation grid profile (for visualization).
    """
    # Step 1: Load checkpoint
    logger.info("Step 1: Loading checkpoint and config...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    data_config = checkpoint["config"]["data"]  # We use this to build dataset class
    data_prep_config = checkpoint["config"]["data_prep"]  # We use this to prepare data

    # Step 2: Prepare the Data (if requested)
    if prepare_data:
        logger.info("Step 2: Preparing Hexel Data...")
        processed_data_dir = prepare_hexel_data(
            data_dir=data_dir,
            hex_id=hex_id,
            win_h=data_prep_config["win_h"],
            win_w=data_prep_config["win_w"],
            overlap_ratio=data_prep_config["overlap_ratio"],
            modelling_approach=data_prep_config["modelling_approach"],
            output_type=data_prep_config["output_type"],
            weather_sampling=data_prep_config["weather_sampling"],
        )
    else:
        processed_data_dir = data_dir / f"data_samples_approach_{data_prep_config['modelling_approach']}"
        logger.info(f"Step 2: Using existing data at {processed_data_dir}")

    # Step 3: Build Dataset
    logger.info("Step 3: Building Dataset...")
    dataset = create_dataset(processed_data_dir, hex_id, data_config)
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(dataset)
    if spatial_channels is None:
        raise ValueError("Could not determine spatial channels from dataset")
    logger.info(f"Dataset: {len(dataset)} samples | Spatial channels: {spatial_channels} | Auxiliary dims: {auxiliary_input_dims}")
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    # Step 4: Instantiate the Predictor (pass pre-loaded checkpoint)
    logger.info("Step 4: Initializing Model Predictor...")
    predictor = BurnRiskPredictor.from_checkpoint(
        checkpoint_path=checkpoint_path, spatial_channels=spatial_channels, auxiliary_input_dims=auxiliary_input_dims
    )

    # Step 5: Run Inference Loop
    logger.info("Step 5: Running Inference...")
    predictions_list = []
    for batch in tqdm(dataloader, desc="Predicting Batches"):
        spatial_inputs = batch["grid"][0]
        batch_preds = predictor(spatial_inputs, auxiliary_inputs=batch)  # Predictor handles device placement internally
        predictions_list.append(batch_preds)

    predictions = torch.cat(predictions_list, dim=0).numpy()
    logger.info(f"Inference complete. Output shape: {predictions.shape}")

    # Step 6: Save patch predictions
    save_pred_path = Path(save_dir) / "predicted_patches" / f"hexel_{hex_id}.npy"
    save_pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(save_pred_path, predictions)
    logger.info(f"Step 6: Saved predictions patches to {save_pred_path}")

    # Step 7: Post-process predictions back to denormalized hexel
    logger.info("Step 7: Post-processing prediction patches into denormalized hexel...")
    grid_source = dataset.sources["grid"]
    reconstructed_hexel_denorm, gt_elevation_grid_profile = get_predicted_hexel(
        base_dir=str(processed_data_dir),
        raw_data_dir=str(data_dir),
        test_df=dataset.metadata,
        predictions=predictions,
        min_target_val=grid_source.BURN_PROB_MIN.item(),  # type: ignore[attr-defined]
        max_target_val=grid_source.BURN_PROB_MAX.item(),  # type: ignore[attr-defined]
        hex_id=hex_id,
    )

    # Step 8: Save reconstructed hexel and visualization
    save_predicted_hexels(
        predicted_hexel=reconstructed_hexel_denorm, hexel_profile=gt_elevation_grid_profile, hex_id=hex_id, save_dir=str(save_dir)
    )
    hex_dir = data_dir / f"hex{hex_id}"
    gt_path = find_simulation_output_file(str(hex_dir), hex_id, output_type=data_prep_config["output_type"])
    gt_grid = load_output_burn_grid(gt_path)
    visualize_burn_prob_grids(gt_grid=gt_grid, pred_grid=reconstructed_hexel_denorm, hex_id=hex_id, save_dir=str(save_dir))
    logger.info(f"Step 8: Saved reconstructed hexel and visualization for hexel {hex_id} in {save_dir}")

    return reconstructed_hexel_denorm, gt_elevation_grid_profile


def main():
    parser = argparse.ArgumentParser(description="Run end-to-end inference on a single hexel.")
    parser.add_argument("--config", type=str, default="inference/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--data_dir", type=str, default=None, help="Directory containing hexel data (overrides config).")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to model checkpoint (overrides config).")
    parser.add_argument("--hex_id", type=str, default=None, help="Hexel ID (overrides config).")
    parser.add_argument("--prepare_data", type=str, default=None, help="Whether to prepare data (overrides config).")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config).")
    parser.add_argument("--num_workers", type=int, default=None, help="Dataloader workers (overrides config).")
    parser.add_argument("--post_process", type=str, default=None, help="Whether to post-process predictions (overrides config).")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save predictions and visualizations (overrides config).")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # CLI args override config (use 'is not None' to allow falsy values like 0)
    data_dir = args.data_dir if args.data_dir is not None else config["data_dir"]
    checkpoint_path = args.checkpoint_path if args.checkpoint_path is not None else config["checkpoint_path"]
    save_dir = args.save_dir if args.save_dir is not None else config["save_dir"]
    hex_id = args.hex_id if args.hex_id is not None else config["hex_id"]
    prepare_data = (args.prepare_data == "True") if args.prepare_data else config["prepare_data"]
    batch_size = args.batch_size if args.batch_size is not None else config["batch_size"]
    num_workers = args.num_workers if args.num_workers is not None else config["num_workers"]

    # Resolve "all" into the list of available hex IDs
    if hex_id == "all":
        hex_ids_to_run = sorted(find_hex_ids(str(Path(config["data_dir"]))))
        logger.info(f"Running inference for all hexels: {hex_ids_to_run}")
    else:
        hex_ids_to_run = [hex_id]

    start_time = time.time()

    for hid in hex_ids_to_run:
        logger.info(f"\n========== Hexel {hid} ==========\n")
        run_single_hexel_pipeline(
            checkpoint_path=Path(config["checkpoint_path"]),
            data_dir=Path(config["data_dir"]),
            hex_id=hid,
            batch_size=batch_size,
            num_workers=num_workers,
            prepare_data=prepare_data,
            save_dir=save_dir,
        )

    elapsed_time = time.time() - start_time
    logger.info(f"Pipeline completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")


if __name__ == "__main__":
    main()
