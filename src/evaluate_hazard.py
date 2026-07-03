"""End-to-end hazard evaluation combining trained BP and FI checkpoints.

python -m src.evaluate_hazard --config configs/hazard_eval_common_input_pipeline.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Collection, Sequence
from typing import Any

import numpy as np
import pandas as pd
import yaml

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_raster, read_split_hex_ids
from data_preparation.utils import find_hex_ids
from src.config import Config, HazardEvalConfig, HazardModelEntry
from src.datasets.dataset import get_test_dataloader
from src.datasets.postprocessing.hazard import compute_raw_hazard, max_finite_hazard
from src.datasets.postprocessing.hazard_metrics import flatten_hazard_class_metrics
from src.datasets.postprocessing.hazard_pipeline import (
    HazardHexelResult,
    compute_hazard_hexel,
    pair_stitched_hexels,
    save_hazard_hexel_artifacts,
)
from src.datasets.postprocessing.hexel_reconstruction import (
    StitchedHexel,
    reconstruct_denormalized_hexels,
)
from src.datasets.postprocessing.utils import get_config_grid_params
from src.datasets.postprocessing.visualize_predictions import as_float_array_with_nan
from src.datasets.targets import get_target_spec
from src.datasets.utils import get_dataset_dimensions
from src.evaluate_hexels import load_config
from src.trainer import Trainer
from src.utils import seed_everything

DENOMINATOR_JSON_FILENAME = "hazard_scale_denominator.json"
PER_HEX_CSV_FILENAME = "hazard_metrics_per_hex.csv"
SUMMARY_JSON_FILENAME = "hazard_metrics_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate combined BP x FI hazard product.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/hazard_eval_common_input_pipeline.yaml",
        help="Path to hazard evaluation YAML config file.",
    )
    parser.add_argument(
        "--metrics_only",
        action="store_true",
        help="Compute hazard metrics without writing GeoTIFF or plot artifacts.",
    )
    parser.add_argument(
        "--skip_plots",
        action="store_true",
        help="Write hazard GeoTIFFs but skip per-hexel PNG plots.",
    )
    parser.add_argument(
        "--no_save_predictions",
        action="store_true",
        help="Override model-entry save_predictions flags and never write patch predictions.",
    )
    return parser.parse_args()


def load_hazard_config(path: str) -> HazardEvalConfig:
    """Load and validate a :class:`HazardEvalConfig` from YAML."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Hazard config file not found: {path}")
    with open(path) as handle:
        raw = yaml.safe_load(handle)
    return HazardEvalConfig(**raw)


def get_model_out_norm(model_config: Config) -> str:
    """Return the grid source ``out_norm`` for a model config, falling back to ``min_max``."""
    grid_params = get_config_grid_params(model_config)
    return grid_params.out_norm if grid_params is not None else "min_max"


def prepare_model_config_for_hazard(
    model_config: Config,
    hazard_config: HazardEvalConfig,
    entry: HazardModelEntry,
    expected_target: str,
) -> Config:
    """Override a model config's data/checkpoint/logger settings for hazard eval.

    Validates the grid target name matches ``expected_target`` (``bp``/``fi``).
    """
    model_config.data.root_dir = hazard_config.root_dir
    model_config.data.raw_data_dir = hazard_config.raw_data_dir
    model_config.data.test_split = hazard_config.test_split
    model_config.data.valid_mask_threshold = hazard_config.valid_mask_threshold
    model_config.evaluation.checkpoint_filename = entry.checkpoint_filename
    model_config.logger.enabled = False

    grid_params = get_config_grid_params(model_config)
    if grid_params is None:
        raise ValueError("Model config has no grid input source; cannot validate hazard target.")
    actual_target = get_target_spec(grid_params.target_name).name
    if actual_target != expected_target:
        raise ValueError(
            f"Expected a {expected_target!r} model, but grid target_name resolves to {actual_target!r} "
            f"(config target_name={grid_params.target_name!r})."
        )
    return model_config


def run_test_inference(model_config: Config, seed: int) -> tuple[np.ndarray, str]:
    """Run test inference; return patch predictions and the grid ``out_norm``."""
    test_loader = get_test_dataloader(
        config=model_config.data,
        modelling_approach=model_config.modelling_approach,
        seed=seed,
    )
    spatial_channels, auxiliary_input_dims = get_dataset_dimensions(test_loader.dataset)
    trainer = Trainer(
        model_config,
        spatial_input_channels=spatial_channels,
        auxiliary_input_dims=auxiliary_input_dims,
    )
    trainer.load_model(filename=model_config.evaluation.checkpoint_filename)
    _, test_predictions = trainer.test(test_loader, return_predictions=True)
    if not isinstance(test_predictions, np.ndarray):
        raise TypeError(f"Expected ndarray predictions, got {type(test_predictions)!r}")
    return test_predictions, get_model_out_norm(model_config)


def read_reference_denominator(path: str) -> float:
    """Read a positive, finite denominator from a JSON reference file.

    Accepts either a bare number or an object with a ``scale_denominator`` or
    ``denominator`` key.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Reference denominator file not found: {path}")
    with open(path) as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        for key in ("scale_denominator", "denominator"):
            if key in payload:
                value = payload[key]
                break
        else:
            raise KeyError(f"Reference denominator file {path!r} is missing a 'scale_denominator' or 'denominator' key.")
    elif isinstance(payload, bool) or not isinstance(payload, (int, float)):
        raise ValueError(
            f"Reference denominator file {path!r} must contain a number or an object with " "'scale_denominator'/'denominator'."
        )
    else:
        value = payload

    numeric = float(value)
    if not np.isfinite(numeric) or numeric <= 0.0:
        raise ValueError(f"reference denominator must be a positive finite number, got {value!r}")
    return numeric


def raw_ground_truth_denominator(
    raw_data_dir: str,
    fi_cap: float | None,
    allowed_hex_ids: Collection[int] | None = None,
) -> float:
    """Compute the max raw hazard over raw BP/FI rasters in ``raw_data_dir``.

    When ``allowed_hex_ids`` is provided, only those hexes contribute (e.g. a
    training split), so the denominator is not derived from held-out data. The
    per-hex maxima are accumulated incrementally to avoid holding every raw
    raster in memory at once.
    """
    hex_ids = find_hex_ids(raw_data_dir)
    if allowed_hex_ids is not None:
        allowed = {int(value) for value in allowed_hex_ids}
        hex_ids = [hex_id for hex_id in hex_ids if int(hex_id) in allowed]
    if not hex_ids:
        raise ValueError(f"No raw hex directories found under {raw_data_dir!r} for the requested hex ids.")

    running_max = -np.inf
    for hex_id in hex_ids:
        paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
        bp_grid = as_float_array_with_nan(load_raster(str(paths.output_burn_prob())))
        fi_grid = as_float_array_with_nan(load_raster(str(paths.output_fire_intensity())))
        raw = compute_raw_hazard(bp_grid, fi_grid, fi_cap)
        finite = raw[np.isfinite(raw)]
        if finite.size:
            running_max = max(running_max, float(finite.max()))
    return max_finite_hazard(np.asarray([running_max]))


def _pairs_raw_hazard_denominator(
    pairs: Sequence[tuple[StitchedHexel, StitchedHexel]],
    fi_cap: float | None,
    use_prediction: bool,
) -> float:
    """Max raw hazard over reconstructed GT (or prediction) BP/FI hexel pairs."""
    if not pairs:
        raise ValueError("No BP/FI hexel pairs available to compute a denominator.")
    raw_grids = [
        compute_raw_hazard(
            bp_hexel.pred_grid if use_prediction else bp_hexel.gt_grid,
            fi_hexel.pred_grid if use_prediction else fi_hexel.gt_grid,
            fi_cap,
        )
        for bp_hexel, fi_hexel in pairs
    ]
    return max_finite_hazard(*raw_grids)


def resolve_hazard_denominator(
    hazard_config: HazardEvalConfig,
    pairs: Sequence[tuple[StitchedHexel, StitchedHexel]],
    *,
    bp_model_config: Config | None = None,
    save_dir: str | None = None,
) -> tuple[float, dict[str, Any]]:
    """Resolve a single scale denominator and return it with metadata.

    An explicit ``scale_denominator`` always wins; otherwise the configured
    ``scale_denominator_source`` determines how the value is derived.
    """
    meta: dict[str, Any] = {"source": hazard_config.scale_denominator_source}

    if hazard_config.scale_denominator is not None:
        denominator = float(hazard_config.scale_denominator)
        meta["source"] = "explicit"
    elif hazard_config.scale_denominator_source == "reference_file":
        if hazard_config.reference_denominator_path is None:
            raise ValueError("reference_denominator_path is required for scale_denominator_source='reference_file'.")
        denominator = read_reference_denominator(hazard_config.reference_denominator_path)
        meta["reference_denominator_path"] = hazard_config.reference_denominator_path
    elif hazard_config.scale_denominator_source == "all_raw_ground_truth":
        denominator = raw_ground_truth_denominator(hazard_config.raw_data_dir, hazard_config.fi_cap)
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            json_path = os.path.join(save_dir, DENOMINATOR_JSON_FILENAME)
            with open(json_path, "w") as handle:
                json.dump({"scale_denominator": denominator, "scale_denominator_source": meta["source"]}, handle, indent=2)
            meta["denominator_json"] = json_path
    elif hazard_config.scale_denominator_source == "train_ground_truth":
        if bp_model_config is None:
            raise ValueError("bp_model_config is required for scale_denominator_source='train_ground_truth'.")
        train_split_path = os.path.join(bp_model_config.data.root_dir, bp_model_config.data.train_split)
        denominator = raw_ground_truth_denominator(hazard_config.raw_data_dir, hazard_config.fi_cap, read_split_hex_ids(train_split_path))
        meta["train_split"] = train_split_path
    elif hazard_config.scale_denominator_source == "eval_ground_truth":
        denominator = _pairs_raw_hazard_denominator(pairs, hazard_config.fi_cap, use_prediction=False)
    elif hazard_config.scale_denominator_source == "prediction":
        denominator = _pairs_raw_hazard_denominator(pairs, hazard_config.fi_cap, use_prediction=True)
    else:  # pragma: no cover - guarded by config validation
        raise ValueError(f"Unsupported scale_denominator_source={hazard_config.scale_denominator_source!r}.")

    denominator = float(denominator)
    meta["value"] = denominator
    return denominator, meta


def write_hazard_metric_summaries(
    results: Sequence[HazardHexelResult],
    save_dir: str,
    denominator: float,
    denominator_metadata: dict[str, Any],
) -> tuple[str, str, dict[str, float]]:
    """Write per-hex CSV and aggregate JSON summaries; return their paths and the aggregate dict."""
    if not results:
        raise ValueError("No hazard hexel results to summarize.")
    os.makedirs(save_dir, exist_ok=True)

    per_hex_df = pd.DataFrame([{"hex_id": result.hex_id, **flatten_hazard_class_metrics(result.metrics)} for result in results])
    csv_path = os.path.join(save_dir, PER_HEX_CSV_FILENAME)
    per_hex_df.to_csv(csv_path, index=False)

    numeric_df = per_hex_df.drop(columns=["hex_id"], errors="ignore").select_dtypes(include=[np.number])
    aggregate: dict[str, float] = {}
    for column in numeric_df.columns:
        values = numeric_df[column].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        aggregate[column] = float(finite.mean()) if finite.size else float("nan")

    json_metrics = {key: (value if np.isfinite(value) else None) for key, value in aggregate.items()}
    summary = {
        "num_hexels": len(results),
        "denominator": float(denominator),
        "denominator_metadata": denominator_metadata,
        "metrics": json_metrics,
    }
    json_path = os.path.join(save_dir, SUMMARY_JSON_FILENAME)
    with open(json_path, "w") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)

    return csv_path, json_path, aggregate


def main() -> None:
    args = parse_args()
    start_time = time.time()
    hazard_config = load_hazard_config(args.config)

    if hazard_config.self_normalized_prediction:
        raise NotImplementedError("self_normalized_prediction is not implemented yet; set it to false in the hazard config.")

    os.makedirs(hazard_config.save_dir, exist_ok=True)

    bp_model_config = prepare_model_config_for_hazard(load_config(hazard_config.bp.config_path), hazard_config, hazard_config.bp, "bp")
    fi_model_config = prepare_model_config_for_hazard(load_config(hazard_config.fi.config_path), hazard_config, hazard_config.fi, "fi")

    seed = getattr(bp_model_config, "seed", 42)
    deterministic = getattr(bp_model_config, "deterministic", True)
    seed_everything(seed=seed, deterministic=deterministic)

    print("\n[Hazard] Running BP test inference...")
    bp_predictions, bp_out_norm = run_test_inference(bp_model_config, seed)
    print("[Hazard] Running FI test inference...")
    fi_predictions, fi_out_norm = run_test_inference(fi_model_config, seed)

    if hazard_config.bp.save_predictions and not args.no_save_predictions:
        np.save(os.path.join(hazard_config.save_dir, "bp_test_predictions.npy"), bp_predictions)
    if hazard_config.fi.save_predictions and not args.no_save_predictions:
        np.save(os.path.join(hazard_config.save_dir, "fi_test_predictions.npy"), fi_predictions)

    bp_hexels = reconstruct_denormalized_hexels(
        test_predictions=bp_predictions,
        config=bp_model_config,
        out_norm=bp_out_norm,
        stitch_mode=hazard_config.stitch_mode,
        mask_scope=hazard_config.mask_scope,
    )
    fi_hexels = reconstruct_denormalized_hexels(
        test_predictions=fi_predictions,
        config=fi_model_config,
        out_norm=fi_out_norm,
        stitch_mode=hazard_config.stitch_mode,
        mask_scope=hazard_config.mask_scope,
    )
    pairs = list(pair_stitched_hexels(bp_hexels, fi_hexels))

    denominator, denominator_metadata = resolve_hazard_denominator(
        hazard_config, pairs, bp_model_config=bp_model_config, save_dir=hazard_config.save_dir
    )
    print(f"[Hazard] Resolved scale denominator={denominator} (source={denominator_metadata['source']})")

    results: list[HazardHexelResult] = []
    for bp_hexel, fi_hexel in pairs:
        result = compute_hazard_hexel(
            bp_hexel,
            fi_hexel,
            denominator=denominator,
            fi_cap=hazard_config.fi_cap,
            scale_to=hazard_config.scale_to,
            bin_thresholds=hazard_config.bin_thresholds,
        )
        results.append(result)
        if not args.metrics_only and hazard_config.save_hazard_map:
            save_hazard_hexel_artifacts(result, hazard_config.save_dir, save_plots=not args.skip_plots)

    _, _, aggregate = write_hazard_metric_summaries(results, hazard_config.save_dir, denominator, denominator_metadata)

    print("\n===== Hazard Evaluation Summary =====")
    print(f"Scale denominator: {denominator} (source={denominator_metadata['source']})")
    print(f"Number of hexels:  {len(results)}")
    print("Aggregate metrics:")
    for key, value in aggregate.items():
        print(f"  {key}: {value:.6f}")
    print(f"===== Total time {round(time.time() - start_time, 3)}s =====")


if __name__ == "__main__":
    main()
