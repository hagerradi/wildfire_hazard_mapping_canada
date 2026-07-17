"""Materializes an edited `weather_table_processed.csv` for one FWI counterfactual scenario."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.counterfactual_weather import apply_fwi_edit, load_all_raw_weather_with_wind_components

WEATHER_INTERVENTION_CSV_NAME = "weather_table_processed.csv"


def weather_intervention_csv_path(prediction_dir: Path) -> Path:
    return prediction_dir / "weather_intervention" / WEATHER_INTERVENTION_CSV_NAME


@dataclass(frozen=True)
class WeatherCounterfactualResult:
    """The edited weather table written for one scenario, plus its edit summary."""

    edited_csv_path: Path
    summary: pd.DataFrame


def materialize_weather_scenario(
    *,
    scenario: ScenarioConfig,
    raw_data_dir: Path,
    processed_weather_csv: Path,
    recipient_hex_ids: list[str],
    prediction_dir: Path,
) -> WeatherCounterfactualResult:
    """Apply `scenario`'s FWI edit and write the resulting table under `prediction_dir`.

    Loads `processed_weather_csv` (the endpoint's shared, unedited `weather_table_processed
    .csv`) and the raw per-hexel weather tables it was built from, applies the scenario's
    edit, and writes the edited table to `weather_intervention_csv_path(prediction_dir)` -
    ready to be pointed at by overriding the endpoint's `spatialized_weather` source
    `csv_name` with an absolute path.
    """
    params = dict(scenario.fwi_edit() or {})
    mode = str(params.pop("mode", "external_extreme_transplant"))

    processed = pd.read_csv(processed_weather_csv)
    raw_features = load_all_raw_weather_with_wind_components(raw_data_dir)
    edited, reports = apply_fwi_edit(
        raw_features,
        processed,
        mode=mode,
        scenario_name=scenario.name,
        recipient_hex_ids=[normalize_hex_id(hex_id) for hex_id in recipient_hex_ids],
        raw_data_dir=raw_data_dir,
        params=params,
    )

    out_path = weather_intervention_csv_path(prediction_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    edited.to_csv(out_path, index=False)
    summary = pd.DataFrame([report.__dict__ for report in reports])
    return WeatherCounterfactualResult(edited_csv_path=out_path, summary=summary)
