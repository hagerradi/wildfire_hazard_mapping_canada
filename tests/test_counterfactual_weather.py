from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.datasets.postprocessing.counterfactual import counterfactual_weather as cw
from src.datasets.postprocessing.counterfactual.counterfactual_base import ScenarioConfig
from src.datasets.postprocessing.counterfactual.weather_counterfactual_transform import (
    materialize_weather_scenario,
    weather_intervention_csv_path,
)


def _weather_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(
        {
            "__hex_id": ["16", "16", "17", "17", "01", "01"],
            "WeatherZone": [4, 9, 30, 31, 1, 1],
            "FireWeatherIndex": [10.0, 20.0, 30.0, 50.0, 5.0, 7.0],
        }
    )
    processed = pd.DataFrame(
        {
            "Order": [1, 2, 3, 4, 5, 6],
            "Season": [1, 2, 1, 2, 1, 2],
            "WeatherZone": raw["WeatherZone"],
            "Temperature": [-1.0, 0.0, 1.0, 3.0, -2.0, -4.0],
            "WindDirection": [180.0, 200.0, 220.0, 240.0, 90.0, 270.0],
            "FireWeatherIndex": [-1.0, 0.0, 2.0, 4.0, -2.0, -4.0],
            "wind_x": [-0.5, -0.25, 0.5, 1.5, 0.0, -1.0],
            "wind_y": [0.0, 0.25, 1.0, 2.0, -0.5, 0.5],
        }
    )
    return raw, processed


def test_apply_external_mean_zone_transplant_builds_exact_recipient_zone_lut() -> None:
    raw, processed = _weather_frames()

    edited, reports = cw.apply_external_mean_zone_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["17"],
        scenario_name="bc_mean_weather_transplant",
    )

    assert len(edited) == processed["WeatherZone"].nunique()
    donor_mean = processed.loc[[2, 3], ["Temperature", "FireWeatherIndex", "wind_x", "wind_y"]].mean()
    for zone in (4, 9):
        row = edited.loc[edited["WeatherZone"] == zone].iloc[0]
        assert row[donor_mean.index].to_numpy(dtype=np.float64) == pytest.approx(donor_mean.to_numpy(dtype=np.float64))

    untouched = edited.loc[edited["WeatherZone"] == 1].iloc[0]
    assert untouched["FireWeatherIndex"] == pytest.approx(processed.loc[[4, 5], "FireWeatherIndex"].mean())
    assert not (set(cw.NON_AVERAGE_COLUMNS) - {"WeatherZone"}).intersection(edited.columns)

    assert len(reports) == 1
    report = reports[0]
    assert report.mode == "external_mean_zone_transplant"
    assert report.recipient_hex_id == "16"
    assert report.n_recipient_rows == 2
    assert report.n_recipient_zones == 2
    assert report.donor_hex_ids == "17"
    assert report.n_donor_rows == 2
    assert report.donor_fwi_mean == pytest.approx(40.0)
    assert report.baseline_fwi_mean == pytest.approx(15.0)
    assert report.scenario_fwi_mean == pytest.approx(40.0)


@pytest.mark.parametrize(
    ("recipient_hex_ids", "donor_hex_ids", "message"),
    [
        (["99"], ["17"], "No weather rows found for recipient"),
        (["16"], ["99"], "No weather rows found for donor"),
        ([], ["17"], "recipient_hex_ids must be non-empty"),
        (["16"], [], "donor_hex_ids must be non-empty"),
    ],
)
def test_apply_external_mean_zone_transplant_validates_hex_ids(
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    message: str,
) -> None:
    raw, processed = _weather_frames()

    with pytest.raises(ValueError, match=message):
        cw.apply_external_mean_zone_transplant(
            raw,
            processed,
            recipient_hex_ids=recipient_hex_ids,
            donor_hex_ids=donor_hex_ids,
            scenario_name="scenario",
        )


def test_apply_external_mean_zone_transplant_rejects_row_count_mismatch() -> None:
    raw, processed = _weather_frames()
    with pytest.raises(ValueError, match="row count mismatch"):
        cw.apply_external_mean_zone_transplant(
            raw.iloc[:-1],
            processed,
            recipient_hex_ids=["16"],
            donor_hex_ids=["17"],
            scenario_name="scenario",
        )


def test_apply_weather_edit_dispatches_mean_mode_and_rejects_invalid_configuration() -> None:
    raw, processed = _weather_frames()
    edited, _ = cw.apply_weather_edit(
        raw,
        processed,
        mode="external_mean_zone_transplant",
        scenario_name="scenario",
        recipient_hex_ids=["16"],
        params={"donor_hex_ids": ["17"]},
    )
    assert set(edited["WeatherZone"]) == set(processed["WeatherZone"])

    with pytest.raises(ValueError, match="Unknown weather edit mode"):
        cw.apply_weather_edit(
            raw,
            processed,
            mode="not_a_mode",
            scenario_name="scenario",
            recipient_hex_ids=["16"],
            params={"donor_hex_ids": ["17"]},
        )
    with pytest.raises(ValueError, match="donor_hex_ids"):
        cw.apply_weather_edit(
            raw,
            processed,
            mode="external_mean_zone_transplant",
            scenario_name="scenario",
            recipient_hex_ids=["16"],
            params={},
        )


def test_weather_scenario_kind_and_weather_edit_accessor() -> None:
    scenario = ScenarioConfig(
        name="bc_mean_weather_transplant",
        kind="weather",
        description="",
        params={"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]},
    )
    assert scenario.weather_edit() == {"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]}
    assert scenario.fuel_edit() is None

    baseline = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
    assert baseline.weather_edit() is None


def test_materialize_mean_weather_scenario_writes_compact_zone_lut(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw, processed = _weather_frames()
    processed_csv = tmp_path / "weather_table_processed.csv"
    processed.to_csv(processed_csv, index=False)
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.weather_counterfactual_transform.load_all_raw_weather_with_hex_ids",
        lambda _raw_data_dir: raw,
    )
    scenario = ScenarioConfig(
        name="bc_mean_weather_transplant",
        kind="weather",
        description="",
        params={"mode": "external_mean_zone_transplant", "donor_hex_ids": ["17"]},
    )
    prediction_dir = tmp_path / "predictions" / scenario.name / "bp"

    result = materialize_weather_scenario(
        scenario=scenario,
        raw_data_dir=tmp_path / "raw",
        processed_weather_csv=processed_csv,
        recipient_hex_ids=["16"],
        prediction_dir=prediction_dir,
    )

    assert result.edited_csv_path == weather_intervention_csv_path(prediction_dir)
    edited = pd.read_csv(result.edited_csv_path)
    assert len(edited) == processed["WeatherZone"].nunique()
    assert edited.loc[edited["WeatherZone"].isin([4, 9]), "FireWeatherIndex"].to_numpy() == pytest.approx([3.0, 3.0])
    assert result.summary[["scenario_name", "donor_hex_ids"]].to_dict("records") == [
        {"scenario_name": "bc_mean_weather_transplant", "donor_hex_ids": "17"}
    ]


def test_materialize_weather_scenario_requires_explicit_mode(tmp_path: Path) -> None:
    processed_csv = tmp_path / "weather_table_processed.csv"
    pd.DataFrame({"WeatherZone": [1], "FireWeatherIndex": [0.0]}).to_csv(processed_csv, index=False)
    scenario = ScenarioConfig(
        name="missing_mode",
        kind="weather",
        description="",
        params={"donor_hex_ids": ["17"]},
    )

    with pytest.raises(ValueError, match="explicit non-empty mode"):
        materialize_weather_scenario(
            scenario=scenario,
            raw_data_dir=tmp_path / "raw",
            processed_weather_csv=processed_csv,
            recipient_hex_ids=["16"],
            prediction_dir=tmp_path / "predictions",
        )
