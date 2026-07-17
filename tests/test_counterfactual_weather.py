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

# Every affine column maps raw -> processed via processed = (raw - 1.0) / 2.0.
_SLOPE = 2.0
_INTERCEPT = 1.0


def _affine_frames(raw_values: dict[str, list[float]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = pd.DataFrame(raw_values)
    processed = pd.DataFrame(
        {column: (np.asarray(values, dtype=np.float64) - _INTERCEPT) / _SLOPE for column, values in raw_values.items()}
    )
    return raw, processed


def test_recover_affine_stats_recovers_known_linear_transform() -> None:
    raw_values = {column: [10.0, 20.0, 30.0, 40.0, 50.0] for column in cw.AFFINE_COLUMNS}
    raw, processed = _affine_frames(raw_values)

    stats = cw.recover_affine_stats(raw, processed, cw.AFFINE_COLUMNS)

    for column in cw.AFFINE_COLUMNS:
        assert stats[column].slope == pytest.approx(_SLOPE)
        assert stats[column].intercept == pytest.approx(_INTERCEPT)


def test_recover_affine_stats_rejects_row_count_mismatch() -> None:
    raw = pd.DataFrame({"Temperature": [1.0, 2.0]})
    processed = pd.DataFrame({"Temperature": [1.0]})
    with pytest.raises(ValueError, match="row count mismatch"):
        cw.recover_affine_stats(raw, processed, ("Temperature",))


def test_recover_affine_stats_rejects_zero_variance_column() -> None:
    raw = pd.DataFrame({"Temperature": [5.0, 5.0, 5.0]})
    processed = pd.DataFrame({"Temperature": [1.0, 1.0, 1.0]})
    with pytest.raises(ValueError, match="zero variance"):
        cw.recover_affine_stats(raw, processed, ("Temperature",))


def test_encode_donor_row_applies_recovered_stats_log1p_and_passthrough() -> None:
    stats = {column: cw.AffineStat(slope=_SLOPE, intercept=_INTERCEPT) for column in cw.AFFINE_COLUMNS}
    donor_row = pd.Series(
        {
            "Order": 5,
            "Season": 1,
            "WeatherZone": 12,
            "Temperature": 30.0,
            "RelativeHumidity": 40.0,
            "WindSpeed": 20.0,
            "WindDirection": 90.0,
            "Precipitation": 3.0,
            "FineFuelMoistureCode": 88.0,
            "DuffMoistureCode": 50.0,
            "DroughtCode": 300.0,
            "InitialSpreadIndex": 12.0,
            "BuildupIndex": 60.0,
            "FireWeatherIndex": 35.0,
        }
    )

    encoded = cw.encode_donor_row(donor_row, stats)

    expected_wind_x, expected_wind_y = cw.wind_to_components(np.array([20.0]), np.array([90.0]))
    for column in cw.AFFINE_COLUMNS:
        if column in ("wind_x", "wind_y"):
            continue
        assert encoded[column] == pytest.approx((float(donor_row[column]) - _INTERCEPT) / _SLOPE)
    assert encoded["wind_x"] == pytest.approx((float(expected_wind_x[0]) - _INTERCEPT) / _SLOPE)
    assert encoded["wind_y"] == pytest.approx((float(expected_wind_y[0]) - _INTERCEPT) / _SLOPE)
    assert encoded["Precipitation"] == pytest.approx(np.log1p(3.0))
    assert encoded["WindDirection"] == pytest.approx(90.0)


def _weather_row(*, order: int, season: int, zone: int, fwi: float) -> dict:
    return {
        "Order": order,
        "Season": season,
        "WeatherZone": zone,
        "Temperature": 20.0,
        "RelativeHumidity": 40.0,
        "WindSpeed": 15.0,
        "WindDirection": 180.0,
        "Precipitation": 0.0,
        "FineFuelMoistureCode": 85.0,
        "DuffMoistureCode": 40.0,
        "DroughtCode": 200.0,
        "InitialSpreadIndex": 8.0,
        "BuildupIndex": 45.0,
        "FireWeatherIndex": fwi,
    }


def test_select_extreme_donor_row_picks_max_fwi_and_respects_season_filter(monkeypatch: pytest.MonkeyPatch) -> None:
    pools = {
        "17": pd.DataFrame([_weather_row(order=1, season=1, zone=30, fwi=25.0), _weather_row(order=2, season=2, zone=30, fwi=60.0)]),
        "02": pd.DataFrame([_weather_row(order=1, season=1, zone=31, fwi=45.0)]),
    }

    def fake_load_weather_list(path: str, season: int | None = None, normalize_weatherlist: bool = True) -> pd.DataFrame:
        for hex_id, frame in pools.items():
            if f"hex{hex_id}" in path:
                return frame.copy()
        raise AssertionError(f"Unexpected weather path: {path}")

    monkeypatch.setattr(cw, "load_weather_list", fake_load_weather_list)

    donor_row, donor_hex_id = cw.select_extreme_donor_row(Path("/raw"), ["17", "02"])
    assert donor_hex_id == "17"
    assert donor_row["FireWeatherIndex"] == pytest.approx(60.0)

    donor_row, donor_hex_id = cw.select_extreme_donor_row(Path("/raw"), ["17", "02"], season_values=[1])
    assert donor_hex_id == "02"
    assert donor_row["FireWeatherIndex"] == pytest.approx(45.0)

    with pytest.raises(ValueError, match="No finite"):
        cw.select_extreme_donor_row(Path("/raw"), ["17", "02"], season_values=[3])


def test_apply_external_extreme_transplant_overwrites_only_recipient_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    raw_values = {column: [10.0, 20.0, 30.0, 40.0] for column in cw.AFFINE_COLUMNS}
    raw, processed = _affine_frames(raw_values)
    raw.insert(0, "__hex_id", ["16", "16", "17", "01"])
    raw["WeatherZone"] = [4, 4, 30, 1]
    raw["WindDirection"] = [180.0, 180.0, 180.0, 180.0]
    raw["Precipitation"] = [0.0, 0.0, 0.0, 0.0]
    processed["WeatherZone"] = raw["WeatherZone"]
    processed["WindDirection"] = raw["WindDirection"]
    processed["Precipitation"] = np.log1p(raw["Precipitation"])

    donor_row = pd.Series(_weather_row(order=9, season=1, zone=52, fwi=99.0))
    monkeypatch.setattr(cw, "select_extreme_donor_row", lambda *_args, **_kwargs: (donor_row, "50"))

    edited, reports = cw.apply_external_extreme_transplant(
        raw,
        processed,
        recipient_hex_ids=["16"],
        donor_hex_ids=["50"],
        raw_data_dir=Path("/raw"),
        scenario_name="bc_extreme_fwi_transplant",
    )

    assert len(reports) == 1
    report = reports[0]
    assert report.recipient_hex_id == "16"
    assert report.n_recipient_rows == 2
    assert report.donor_hex_id == "50"
    assert report.donor_fwi == pytest.approx(99.0)

    expected_fwi = cw.encode_donor_row(donor_row, cw.recover_affine_stats(raw, processed, cw.AFFINE_COLUMNS))["FireWeatherIndex"]
    assert edited.loc[[0, 1], "FireWeatherIndex"].to_numpy() == pytest.approx([expected_fwi, expected_fwi])
    # Non-recipient rows are untouched.
    assert edited.loc[2, "FireWeatherIndex"] == pytest.approx(processed.loc[2, "FireWeatherIndex"])
    assert edited.loc[3, "FireWeatherIndex"] == pytest.approx(processed.loc[3, "FireWeatherIndex"])


def test_apply_external_extreme_transplant_raises_for_unknown_recipient() -> None:
    raw_values = {column: [10.0, 20.0] for column in cw.AFFINE_COLUMNS}
    raw, processed = _affine_frames(raw_values)
    raw.insert(0, "__hex_id", ["16", "16"])

    with pytest.raises(ValueError, match="No weather rows found for recipient"):
        cw.apply_external_extreme_transplant(
            raw,
            processed,
            recipient_hex_ids=["99"],
            donor_hex_ids=["50"],
            raw_data_dir=Path("/raw"),
            scenario_name="scenario",
        )


def test_apply_fwi_edit_rejects_unknown_mode_and_missing_donor_hex_ids() -> None:
    raw = pd.DataFrame({"__hex_id": ["16"]})
    processed = pd.DataFrame({"FireWeatherIndex": [1.0]})

    with pytest.raises(ValueError, match="Unknown weather edit mode"):
        cw.apply_fwi_edit(
            raw, processed, mode="not_a_mode", scenario_name="s", recipient_hex_ids=["16"], raw_data_dir=Path("/raw"), params={}
        )

    with pytest.raises(ValueError, match="donor_hex_ids"):
        cw.apply_fwi_edit(
            raw,
            processed,
            mode="external_extreme_transplant",
            scenario_name="s",
            recipient_hex_ids=["16"],
            raw_data_dir=Path("/raw"),
            params={},
        )


def test_fwi_scenario_kind_and_fwi_edit_accessor() -> None:
    scenario = ScenarioConfig(
        name="bc_extreme_fwi_transplant",
        kind="fwi",
        description="",
        params={"mode": "external_extreme_transplant", "donor_hex_ids": ["17"]},
    )
    assert scenario.fwi_edit() == {"mode": "external_extreme_transplant", "donor_hex_ids": ["17"]}
    assert scenario.fuel_edit() is None

    baseline = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
    assert baseline.fwi_edit() is None


def test_materialize_weather_scenario_writes_edited_csv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    processed_csv = tmp_path / "weather_table_processed.csv"
    raw_values = {column: [10.0, 20.0, 30.0] for column in cw.AFFINE_COLUMNS}
    _, processed = _affine_frames(raw_values)
    processed["WeatherZone"] = [4, 4, 30]
    processed["WindDirection"] = [180.0, 180.0, 180.0]
    processed["Precipitation"] = [0.0, 0.0, 0.0]
    processed.to_csv(processed_csv, index=False)

    raw_features = pd.DataFrame(raw_values)
    raw_features.insert(0, "__hex_id", ["16", "16", "17"])
    raw_features["WeatherZone"] = processed["WeatherZone"]
    raw_features["WindDirection"] = processed["WindDirection"]
    raw_features["Precipitation"] = processed["Precipitation"]
    monkeypatch.setattr(
        "src.datasets.postprocessing.counterfactual.weather_counterfactual_transform.load_all_raw_weather_with_wind_components",
        lambda _raw_data_dir: raw_features,
    )
    donor_row = pd.Series(_weather_row(order=1, season=1, zone=52, fwi=99.0))
    monkeypatch.setattr(cw, "select_extreme_donor_row", lambda *_args, **_kwargs: (donor_row, "50"))

    scenario = ScenarioConfig(
        name="bc_extreme_fwi_transplant",
        kind="fwi",
        description="",
        params={"mode": "external_extreme_transplant", "donor_hex_ids": ["50"]},
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
    assert result.edited_csv_path.exists()
    edited = pd.read_csv(result.edited_csv_path)
    assert edited.loc[[0, 1], "FireWeatherIndex"].nunique() == 1
    assert edited.loc[2, "FireWeatherIndex"] == pytest.approx(processed.loc[2, "FireWeatherIndex"])
    assert list(result.summary["recipient_hex_id"]) == ["16"]
