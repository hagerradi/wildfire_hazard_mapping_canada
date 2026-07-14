"""Run fixed checkpoints against baseline and counterfactual patch transforms."""

from __future__ import annotations

import argparse
import os
import shutil
from functools import partial
from pathlib import Path

import pandas as pd

from src.datasets.fuel_counterfactual import FuelCounterfactualTransform
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual import ScenarioConfig, load_counterfactual_config
from src.evaluate_hexels import load_config
from src.evaluate_hexels import main as evaluate_hexels


def _resolve_path(path: Path | str, project_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else project_root / value


def _select_scenarios(scenarios: list[ScenarioConfig], scenario_names: set[str] | None) -> list[ScenarioConfig]:
    """Filter to the requested scenarios, always keeping the baseline scenario.

    Downstream plotting scripts diff every fuel scenario against baseline, so baseline
    predictions must exist even if `scenario_names` doesn't request it explicitly.
    """
    if scenario_names is None:
        return list(scenarios)
    return [scenario for scenario in scenarios if scenario.kind == "baseline" or scenario.name in scenario_names]


def _filter_hex_ids(metadata: pd.DataFrame, hex_ids: set[str]) -> pd.DataFrame:
    normalized = metadata["hex_id"].astype(str).map(normalize_hex_id)
    return metadata.loc[normalized.isin(hex_ids)]


def _fuel_channel(data_root: Path, modelling_approach: str) -> int:
    import json

    path = data_root / f"feature_channel_map_{modelling_approach}.json"
    with path.open() as handle:
        channel_map = json.load(handle)
    channels = channel_map.get("fuel_grid")
    if not channels:
        raise ValueError(f"{path} has no fuel_grid channel.")
    return int(channels[0])


def _prepare_prediction_dir(
    *,
    source_save_dir: Path,
    prediction_dir: Path,
    checkpoint_filename: str,
    overwrite: bool,
) -> None:
    if prediction_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{prediction_dir} already exists; pass --overwrite to replace it.")
        shutil.rmtree(prediction_dir)
    prediction_dir.mkdir(parents=True)
    source_checkpoint = source_save_dir / checkpoint_filename
    destination_checkpoint = prediction_dir / checkpoint_filename
    if not source_checkpoint.exists():
        raise FileNotFoundError(source_checkpoint)
    try:
        os.link(source_checkpoint, destination_checkpoint)
    except OSError:
        shutil.copy2(source_checkpoint, destination_checkpoint)


def _evaluation_args() -> argparse.Namespace:
    return argparse.Namespace(
        config="",
        visualize_predictions=False,
        save_visualizations=False,
        metrics_only=False,
        skip_hexel_plots=True,
        no_save_predictions=True,
        robust_plot_percentile=None,
        stitch_mode="mean",
        mask_scope="actual",
    )


def run_counterfactual_evaluation(
    config_path: Path,
    *,
    endpoint_names: set[str] | None = None,
    scenario_names: set[str] | None = None,
    overwrite: bool = False,
    project_root: Path | None = None,
) -> pd.DataFrame:
    """Evaluate each selected endpoint under each selected scenario for the configured hexels.

    For every (endpoint, scenario) pair, this loads the endpoint's checkpoint config,
    filters test metadata to `config.hex_ids`, applies the scenario's fuel-edit patch
    transform (a no-op for the baseline scenario), and runs `evaluate_hexels` to write
    predicted hexel rasters under `<save_dir>/predictions/<scenario>/<endpoint>/`. Also
    writes, under `save_dir`: `scenario_prediction_index.csv` (returned), `counterfactual
    _metrics.csv`, and (for fuel scenarios) `fuel_edit_summary.csv` / `fuel_component_
    replacements.csv`.

    The baseline scenario is always evaluated regardless of `scenario_names`, since
    downstream plotting scripts diff each fuel scenario against it.

    Returns:
        The scenario/endpoint -> prediction_dir index, as also written to
        `scenario_prediction_index.csv`.
    """
    project_root = (project_root or Path.cwd()).resolve()
    config = load_counterfactual_config(config_path)
    save_dir = _resolve_path(config.save_dir, project_root)
    raw_data_dir = _resolve_path(config.raw_data_dir, project_root)
    hex_ids = set(config.hex_ids)
    endpoints = [endpoint for endpoint in config.endpoints.values() if endpoint_names is None or endpoint.name in endpoint_names]
    scenarios = _select_scenarios(config.scenarios, scenario_names)
    if not endpoints:
        raise ValueError("No enabled endpoints selected.")
    if not scenarios:
        raise ValueError("No scenarios selected.")

    index_rows = []
    metric_rows: list[dict[str, str | float]] = []
    summary_frames = []
    component_frames = []
    for endpoint in endpoints:
        endpoint_config_path = _resolve_path(endpoint.config_path, project_root)
        base_config = load_config(str(endpoint_config_path))
        source_save_dir = _resolve_path(base_config.save_dir, project_root)
        data_root = _resolve_path(endpoint.baseline_data_root or base_config.data.root_dir, project_root)
        metadata = pd.read_csv(data_root / base_config.data.test_split)
        if "valid_ratio" in metadata.columns:
            metadata = metadata.loc[metadata["valid_ratio"] > base_config.data.valid_mask_threshold]
        metadata = _filter_hex_ids(metadata, hex_ids).reset_index(drop=True)
        if metadata.empty:
            raise ValueError(f"No test metadata found for hex_ids={sorted(hex_ids)} and endpoint={endpoint.name!r}.")

        for scenario in scenarios:
            prediction_dir = save_dir / "predictions" / scenario.name / endpoint.name
            checkpoint_filename = base_config.evaluation.checkpoint_filename
            _prepare_prediction_dir(
                source_save_dir=source_save_dir,
                prediction_dir=prediction_dir,
                checkpoint_filename=checkpoint_filename,
                overwrite=overwrite,
            )

            run_config = base_config.model_copy(deep=True)
            run_config.save_dir = str(prediction_dir)
            run_config.data.root_dir = str(data_root)
            run_config.data.raw_data_dir = str(raw_data_dir)
            run_config.data.num_workers = 0
            run_config.logger.enabled = False

            patch_transform = None
            if scenario.kind == "fuel":
                patch_transform = FuelCounterfactualTransform.from_metadata(
                    data_root=data_root,
                    metadata=metadata,
                    fuel_channel=_fuel_channel(data_root, run_config.modelling_approach),
                    scenario=scenario,
                    filename_col=base_config.data.filename_col,
                )
                summary = patch_transform.summary.copy()
                summary.insert(0, "endpoint", endpoint.name)
                summary_frames.append(summary)
                if not patch_transform.components.empty:
                    components = patch_transform.components.copy()
                    components.insert(0, "endpoint", endpoint.name)
                    component_frames.append(components)

            metrics = evaluate_hexels(
                args=_evaluation_args(),
                config=run_config,
                patch_transform=patch_transform,
                metadata_filter=partial(_filter_hex_ids, hex_ids=hex_ids),
            )
            metric_rows.extend(
                {
                    "scenario": scenario.name,
                    "endpoint": endpoint.name,
                    "metric": metric,
                    "value": value,
                }
                for metric, value in metrics.items()
            )
            index_rows.append(
                {
                    "scenario": scenario.name,
                    "endpoint": endpoint.name,
                    "prediction_dir": str(prediction_dir.resolve()),
                }
            )

    save_dir.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(index_rows)
    index.to_csv(save_dir / "scenario_prediction_index.csv", index=False)
    if metric_rows:
        pd.DataFrame(metric_rows).to_csv(save_dir / "counterfactual_metrics.csv", index=False)
    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(save_dir / "fuel_edit_summary.csv", index=False)
    if component_frames:
        pd.concat(component_frames, ignore_index=True).to_csv(save_dir / "fuel_component_replacements.csv", index=False)
    return index


def _parse_name_set(values: list[str] | None) -> set[str] | None:
    return set(values) if values else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/counterfactual_fuel.yaml"))
    parser.add_argument("--endpoint", action="append", dest="endpoints")
    parser.add_argument("--scenario", action="append", dest="scenarios")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index = run_counterfactual_evaluation(
        args.config,
        endpoint_names=_parse_name_set(args.endpoints),
        scenario_names=_parse_name_set(args.scenarios),
        overwrite=args.overwrite,
    )
    print(f"Completed {len(index)} counterfactual evaluations.")


if __name__ == "__main__":
    main()
