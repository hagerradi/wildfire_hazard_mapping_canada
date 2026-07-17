"""Raw weather-table editing helpers for FWI/severe-weather counterfactual analyses.

`weather_table_processed.csv` is built once, upstream of any counterfactual run, by
aggregating every hexel's raw `hex{NN}_DailyWeather.csv` (in sorted-glob order) and
applying column-wise scaling (`data_preparation.tabular.weather.preprocess_weather_list`):
min-max for RelativeHumidity/FineFuelMoistureCode, z-score for the remaining thermo/wind
columns (including the derived `wind_x`/`wind_y`), and `log1p` for Precipitation. Neither
the fitted scalers nor a hex-id column are persisted alongside it.

To transplant a donor weather row into a recipient hexel's rows without touching that
upstream pipeline, this module:
1. Reconstructs the raw (unscaled) aggregated table in the same row order, tagged with
   each row's hex id, and recovers each scaled column's affine transform by pairing it
   against the persisted processed table (`recover_affine_stats`) - this works because
   every persisted transform (min-max or z-score) is affine in the raw value.
2. Selects a donor row - e.g. the most extreme `FireWeatherIndex` day across a set of
   donor hexels (`select_extreme_donor_row`) - from those hexels' own raw weather tables.
3. Encodes that donor row through the recovered transforms (`encode_donor_row`) and writes
   it over every recipient row (`apply_external_extreme_transplant`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import Paths
from data_preparation.tabular.weather import load_weather_list, wind_to_components
from data_preparation.utils import aggregate_csv_by_pattern
from src.datasets.fuel_utils import normalize_hex_id

FWI_COLUMN = "FireWeatherIndex"
STRUCTURAL_COLUMNS: tuple[str, ...] = ("Order", "Season", "WeatherZone")
PASSTHROUGH_COLUMNS: tuple[str, ...] = ("WindDirection",)
LOG1P_COLUMNS: tuple[str, ...] = ("Precipitation",)
AFFINE_COLUMNS: tuple[str, ...] = (
    "Temperature",
    "RelativeHumidity",
    "WindSpeed",
    "FineFuelMoistureCode",
    "DuffMoistureCode",
    "DroughtCode",
    "InitialSpreadIndex",
    "BuildupIndex",
    FWI_COLUMN,
    "wind_x",
    "wind_y",
)
WEATHER_EDIT_MODES = ("external_extreme_transplant",)
RAW_WEATHER_GLOB_PATTERN = "hex*/tabular/hex*_DailyWeather.csv"


@dataclass(frozen=True)
class AffineStat:
    """One recovered affine transform: raw = slope * processed + intercept."""

    slope: float
    intercept: float


@dataclass(frozen=True)
class WeatherEditReport:
    """Summary of one FWI/severe-weather counterfactual edit."""

    scenario_name: str
    mode: str
    recipient_hex_id: str
    n_recipient_rows: int
    n_recipient_zones: int
    donor_hex_id: str
    donor_order: int
    donor_season: int
    donor_zone: int
    donor_fwi: float
    baseline_fwi_mean: float
    scenario_fwi_mean: float
    note: str

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([asdict(self)])


def _hex_id_from_weather_path(path: Path) -> str:
    return normalize_hex_id(path.parent.parent.name.removeprefix("hex"))


def _load_raw_weather_with_hex_id(path: Path) -> pd.DataFrame:
    df = load_weather_list(str(path), normalize_weatherlist=False)
    df.insert(0, "__hex_id", _hex_id_from_weather_path(path))
    return df


def load_all_raw_weather_with_wind_components(raw_data_dir: Path) -> pd.DataFrame:
    """Reconstruct the unscaled, hex-id-tagged weather table used to build the processed table.

    Rows are produced in the same sorted-glob order as `build_weather_table`
    (`data_preparation/process_tabular_data.py`), so this table's rows line up
    one-to-one with `weather_table_processed.csv`'s rows.
    """
    raw = aggregate_csv_by_pattern(root_dir=raw_data_dir, pattern=RAW_WEATHER_GLOB_PATTERN, load_function=_load_raw_weather_with_hex_id)
    wind_x, wind_y = wind_to_components(raw["WindSpeed"].to_numpy(dtype=np.float64), raw["WindDirection"].to_numpy(dtype=np.float64))
    return raw.assign(wind_x=wind_x, wind_y=wind_y)


def recover_affine_stats(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    columns: tuple[str, ...] = AFFINE_COLUMNS,
) -> dict[str, AffineStat]:
    """Recover each column's affine scaling by pairing raw and processed rows.

    Both min-max and z-score scaling are affine in the raw value (`processed = (raw -
    intercept) / slope`), so a single linear-regression recovery covers every scaled
    weather column without needing to know which scaling each one used.
    """
    if len(raw_features) != len(processed):
        raise ValueError(f"Raw/processed row count mismatch: {len(raw_features)} != {len(processed)}.")

    stats: dict[str, AffineStat] = {}
    for column in columns:
        if column not in raw_features.columns:
            raise ValueError(f"Raw weather features are missing column {column!r}.")
        if column not in processed.columns:
            raise ValueError(f"Processed weather table is missing column {column!r}.")

        x = processed[column].to_numpy(dtype=np.float64)
        y = raw_features[column].to_numpy(dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        if int(finite.sum()) < 2:
            raise ValueError(f"Need at least two finite rows to recover a transform for {column!r}.")

        var_x = float(np.var(x[finite]))
        if var_x <= 0.0:
            raise ValueError(f"Processed column {column!r} has zero variance; cannot recover its scaling.")
        slope = float(np.cov(x[finite], y[finite], bias=True)[0, 1] / var_x)
        intercept = float(np.mean(y[finite]) - slope * np.mean(x[finite]))
        if not np.isfinite(slope) or not np.isfinite(intercept) or slope == 0.0:
            raise ValueError(f"Invalid recovered transform for {column!r}: slope={slope}, intercept={intercept}.")
        stats[column] = AffineStat(slope=slope, intercept=intercept)
    return stats


def encode_donor_row(donor_row: pd.Series, stats: dict[str, AffineStat]) -> dict[str, float]:
    """Encode one raw donor weather row through the recovered processed-table transforms."""
    wind_x, wind_y = wind_to_components(
        np.array([donor_row["WindSpeed"]], dtype=np.float64),
        np.array([donor_row["WindDirection"]], dtype=np.float64),
    )
    raw_values = {**donor_row.to_dict(), "wind_x": float(wind_x[0]), "wind_y": float(wind_y[0])}

    encoded: dict[str, float] = {}
    for column in AFFINE_COLUMNS:
        stat = stats[column]
        encoded[column] = (float(raw_values[column]) - stat.intercept) / stat.slope
    for column in LOG1P_COLUMNS:
        encoded[column] = float(np.log1p(raw_values[column]))
    for column in PASSTHROUGH_COLUMNS:
        encoded[column] = float(raw_values[column])
    return encoded


def _normalize_season_values(values: object) -> set[int] | None:
    if values is None:
        return None
    raw_values = values if isinstance(values, list | tuple | set) else [values]
    normalized: set[int] = set()
    for value in raw_values:
        text = str(value).strip()
        if text.lower().startswith("s"):
            text = text[1:]
        if not text:
            raise ValueError("season_values cannot include empty values.")
        normalized.add(int(float(text)))
    if not normalized:
        raise ValueError("season_values must include at least one season.")
    return normalized


def select_extreme_donor_row(
    raw_data_dir: Path,
    donor_hex_ids: list[str],
    *,
    rank_column: str = FWI_COLUMN,
    season_values: object = None,
) -> tuple[pd.Series, str]:
    """Return the highest-`rank_column` raw weather row from `donor_hex_ids`, and its hex id."""
    if not donor_hex_ids:
        raise ValueError("donor_hex_ids must be non-empty.")

    frames = []
    for hex_id in donor_hex_ids:
        normalized_hex_id = normalize_hex_id(hex_id)
        path = Paths(hex_id=normalized_hex_id, root_dir=raw_data_dir).weather_table(normalized_hex_id)
        df = load_weather_list(str(path), normalize_weatherlist=False)
        df.insert(0, "__hex_id", normalized_hex_id)
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)

    allowed_seasons = _normalize_season_values(season_values)
    if allowed_seasons is not None:
        pool = pool.loc[pool["Season"].isin(allowed_seasons)]

    rank_values = pool[rank_column].to_numpy(dtype=np.float64)
    finite_positions = np.flatnonzero(np.isfinite(rank_values))
    if finite_positions.size == 0:
        season_suffix = f" in seasons {sorted(allowed_seasons)}" if allowed_seasons is not None else ""
        raise ValueError(f"No finite {rank_column!r} donor rows found for donor_hex_ids={donor_hex_ids}{season_suffix}.")

    donor_position = int(finite_positions[int(np.argmax(rank_values[finite_positions]))])
    donor_row = pool.iloc[donor_position]
    return donor_row, str(donor_row["__hex_id"])


def apply_external_extreme_transplant(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    recipient_hex_ids: list[str],
    donor_hex_ids: list[str],
    raw_data_dir: Path,
    scenario_name: str,
    rank_column: str = FWI_COLUMN,
    season_values: object = None,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Overwrite every recipient-hexel weather row with one extreme donor-hexel row.

    `raw_features` and `processed` must be row-aligned (see
    `load_all_raw_weather_with_wind_components`); `raw_features["__hex_id"]` selects each
    recipient's rows within `processed`. Every recipient hexel receives the same donor row.
    """
    if "__hex_id" not in raw_features.columns:
        raise ValueError("raw_features is missing the '__hex_id' tag column.")
    normalized_recipient_ids = [normalize_hex_id(hex_id) for hex_id in recipient_hex_ids]
    recipient_masks = {hex_id: (raw_features["__hex_id"] == hex_id).to_numpy() for hex_id in normalized_recipient_ids}
    missing_recipients = [hex_id for hex_id, mask in recipient_masks.items() if not mask.any()]
    if missing_recipients:
        raise ValueError(f"No weather rows found for recipient hex_id(s)={missing_recipients}.")

    stats = recover_affine_stats(raw_features, processed, AFFINE_COLUMNS)
    donor_row, donor_hex_id = select_extreme_donor_row(raw_data_dir, donor_hex_ids, rank_column=rank_column, season_values=season_values)
    encoded_donor = encode_donor_row(donor_row, stats)

    edited = processed.copy()
    reports: list[WeatherEditReport] = []
    for recipient_hex_id in normalized_recipient_ids:
        recipient_mask = recipient_masks[recipient_hex_id]
        recipient_positions = np.flatnonzero(recipient_mask)
        for column, value in encoded_donor.items():
            edited.loc[recipient_positions, column] = value

        reports.append(
            WeatherEditReport(
                scenario_name=scenario_name,
                mode="external_extreme_transplant",
                recipient_hex_id=recipient_hex_id,
                n_recipient_rows=int(recipient_positions.size),
                n_recipient_zones=int(raw_features.loc[recipient_mask, "WeatherZone"].nunique()),
                donor_hex_id=donor_hex_id,
                donor_order=int(donor_row["Order"]),
                donor_season=int(donor_row["Season"]),
                donor_zone=int(donor_row["WeatherZone"]),
                donor_fwi=float(donor_row[rank_column]),
                baseline_fwi_mean=float(raw_features.loc[recipient_mask, FWI_COLUMN].mean()),
                scenario_fwi_mean=float(donor_row[FWI_COLUMN]),
                note="ok",
            )
        )
    return edited, reports


def apply_fwi_edit(
    raw_features: pd.DataFrame,
    processed: pd.DataFrame,
    *,
    mode: str,
    scenario_name: str,
    recipient_hex_ids: list[str],
    raw_data_dir: Path,
    params: dict,
) -> tuple[pd.DataFrame, list[WeatherEditReport]]:
    """Dispatch one FWI/severe-weather counterfactual edit by `mode`."""
    if mode not in WEATHER_EDIT_MODES:
        raise ValueError(f"Unknown weather edit mode {mode!r}; expected one of {WEATHER_EDIT_MODES}.")
    donor_hex_ids = params.get("donor_hex_ids")
    if not isinstance(donor_hex_ids, list | tuple) or not donor_hex_ids:
        raise ValueError(f"Weather scenario {scenario_name!r} must define a non-empty donor_hex_ids list.")
    return apply_external_extreme_transplant(
        raw_features,
        processed,
        recipient_hex_ids=recipient_hex_ids,
        donor_hex_ids=[str(value) for value in donor_hex_ids],
        raw_data_dir=raw_data_dir,
        scenario_name=scenario_name,
        rank_column=str(params.get("rank_column", FWI_COLUMN)),
        season_values=params.get("season_values"),
    )
