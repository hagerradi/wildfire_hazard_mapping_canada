# Counterfactual Extreme-Weather (FWI) Intervention

Evaluate how model predictions (BP, FI, ROS) respond to a hypothetical severe-weather
day — "what if hex16 experienced the most extreme fire-weather day observed in nearby
BC-area hexels?" — for a fixed set of hexels, using already-trained checkpoints.

This reuses the fuel-intervention pipeline's structure (see
`COUNTERFACTUAL_FUEL_README.md`): the same evaluation entry point
(`src/evaluate_counterfactual.py`) and the same generic response-map plotting script
(`counterfactual_response_maps.py`), with a weather-specific edit step in place of the
fuel patch transform.

## Pipeline

```
configs/counterfactual_extreme_weather.yaml
        │
        ▼
src/evaluate_counterfactual.py                     # 1. Evaluate baseline + selected scenario(s), per endpoint
        │
        ▼
src/datasets/postprocessing/counterfactual/
  counterfactual_weather.py                        #    Raw weather-table editing helpers (donor selection, encoding)
  weather_counterfactual_transform.py               #    Writes the edited weather_table_processed.csv
  plotting/counterfactual_response_maps.py          # 2. Plot GT/baseline/scenario/Δ prediction maps + hotspot patch zoom
```

1. **Evaluate** (`evaluate_counterfactual.py`): runs the `baseline` scenario (unmodified
   weather) and each selected `fwi` scenario, for each selected endpoint. `fwi` scenarios
   materialize an edited `weather_table_processed.csv` (see below) and point that
   endpoint's `spatialized_weather` input source at it via an absolute `csv_name` path -
   `SpatializedTabularSource` resolves its CSV as `os.path.join(root_dir, csv_name)`,
   which returns an absolute `csv_name` unchanged, so this doesn't require duplicating the
   rest of the data root. Writes predicted hexel rasters under
   `<save_dir>/predictions/<scenario>/<endpoint>/`, an index mapping
   `(scenario, endpoint) -> prediction_dir` (`scenario_prediction_index.csv`), evaluation
   metrics (`counterfactual_metrics.csv`), and a weather-edit summary
   (`weather_edit_summary.csv`). Each prediction directory's `weather_intervention/`
   subdirectory holds the exact edited `weather_table_processed.csv` used by inference.
2. **Response maps** (`counterfactual_response_maps.py`): generic, endpoint-parameterized
   (`--endpoint {bp,fi,ros}`) ground-truth/baseline/scenario/Δ maps for one hexel/scenario
   pair, plus zoom-ins on the highest-Δ patches and per-pixel Δ histogram/concentration
   plots. Shared as-is with the fuel-intervention pipeline.

## How the weather edit works

`weather_table_processed.csv` is built once, upstream of any counterfactual run, from
every hexel's raw `hex{NN}_DailyWeather.csv` (see `data_preparation/process_tabular_data
.py`): min-max scaling for RelativeHumidity/FineFuelMoistureCode, z-score scaling for the
remaining thermo/wind columns (including the derived `wind_x`/`wind_y`), and `log1p` for
Precipitation. Neither the fitted scalers nor a hex-id column are persisted alongside it.

To transplant a donor weather row into a recipient hexel's rows without touching that
upstream pipeline (`counterfactual_weather.py`):

1. Reconstruct the raw (unscaled) aggregated table in the same row order used to build
   `weather_table_processed.csv`, tagged with each row's hex id
   (`load_all_raw_weather_with_wind_components`).
2. Recover each scaled column's affine transform (`recover_affine_stats`) by pairing it
   against the persisted processed table - every persisted transform (min-max or
   z-score) is affine in the raw value, so this works without knowing which scaling any
   given column used.
3. Select a donor row - the highest-`FireWeatherIndex` day across a configured set of
   donor hexels (`select_extreme_donor_row`) - from those hexels' own raw weather tables.
4. Encode that donor row through the recovered transforms (`encode_donor_row`) and write
   it over every recipient row (`apply_external_extreme_transplant`).

## Configuration (`configs/counterfactual_extreme_weather.yaml`)

```yaml
raw_data_dir: "/path/to/raw/hexel/data"
save_dir: "experiments/counterfactual_extreme_weather_hex16"
hex_ids: ["16"]

endpoints:
  bp:
    config_path: "configs/bp_common_input_pipeline.yaml"
  fi:
    config_path: "configs/fi_common_input_pipeline.yaml"
  ros:
    config_path: "configs/ros_common_input_pipeline.yaml"

scenarios:
  - name: "baseline"
    kind: "baseline"
    description: "Unmodified prepared weather inputs."

  - name: "bc_extreme_fwi_transplant"
    kind: "fwi"
    description: "Replace every hex16 weather row with the single highest-FireWeatherIndex day found across configured BC-area donor hexels."
    params:
      mode: "external_extreme_transplant"
      rank_column: "FireWeatherIndex"
      donor_hex_ids: ["17", "02", "19", "25", "26", "27", "40", "50", "52"]
```

- `endpoints`: one entry per trained model to evaluate; `config_path` points at that
  model's own training/evaluation config (used to resolve its checkpoint, data root, and
  test split). Each endpoint's config must have a `spatialized_weather` input source.
- `scenarios`: exactly one scenario named `baseline` with `kind: "baseline"`, plus any
  number of `fwi` scenarios. Each `fwi` scenario's `params` are passed to `apply_fwi_edit`
  (`counterfactual_weather.py`), keyed by `mode`:

  | `mode` | Required params | Optional params | Behaviour |
  |---|---|---|---|
  | `external_extreme_transplant` | `donor_hex_ids` | `rank_column` (default `FireWeatherIndex`), `season_values` | Every recipient hexel's (`hex_ids`) weather row is overwritten with the single highest-`rank_column` day found across `donor_hex_ids`'s own raw weather tables, optionally restricted to `season_values` (e.g. `[1]` for spring/leaf-off). |

  Adding a new scenario `mode` means adding a branch in `apply_fwi_edit` and a matching
  entry in this table.

## Running

By default, `evaluate_counterfactual.py` runs every endpoint and scenario in the config.
Pass `--endpoint`/`--scenario` (repeatable) to restrict to a subset:

```bash
# Evaluate everything in the config
python -m src.evaluate_counterfactual --config configs/counterfactual_extreme_weather.yaml --overwrite

# Evaluate only the fi endpoint for the BC extreme-FWI transplant (baseline is always included)
python -m src.evaluate_counterfactual --config configs/counterfactual_extreme_weather.yaml \
    --endpoint fi --scenario bc_extreme_fwi_transplant
```

Plotting scripts operate on one hexel/scenario/endpoint at a time and read paths from the
counterfactual config plus the `scenario_prediction_index.csv` written by evaluation.
`--experiment_dir` and `--raw_data_dir` are optional overrides:

```bash
# Plot GT/baseline/scenario/Δ response maps + patch zoom for one endpoint
python -m src.datasets.postprocessing.counterfactual.plotting.counterfactual_response_maps \
    --config configs/counterfactual_extreme_weather.yaml \
    --scenario bc_extreme_fwi_transplant --endpoint fi --hex_id 16
```

Run `--help` on any script for the full set of options (e.g. `--zone_overlay` to draw
firezone boundaries, `--downsample` for lower-resolution map rendering).

Submit `run_files/counterfactual_extreme_weather_iROS.sh` for GPU evaluation, then
`run_files/counterfactual_extreme_weather_plots.sh` for the configured response plots.
