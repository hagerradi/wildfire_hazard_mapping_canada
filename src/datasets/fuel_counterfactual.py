"""In-memory fuel counterfactual transform for prepared patch datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import rasterio

from data_preparation.paths import Paths
from data_preparation.spatial.utils import load_spatial_raster
from src.datasets.fuel_utils import normalize_hex_id
from src.datasets.postprocessing.counterfactual import ScenarioConfig
from src.datasets.postprocessing.counterfactual_fuel import FUEL_NODATA, apply_fuel_edit
from src.datasets.postprocessing.stitch_hexel import stitch_windows


@dataclass(frozen=True)
class PatchFuelWindow:
    hex_id: str
    row: int
    col: int
    shape: tuple[int, int]


def fuel_intervention_raster_path(
    prediction_dir: Path,
    hex_id: str,
    variant: Literal["baseline", "scenario"],
) -> Path:
    return prediction_dir / "fuel_intervention" / f"hexel_{int(hex_id):02d}_{variant}_fuel.tif"


def _write_fuel_raster(data: np.ndarray, profile: dict, path: Path) -> None:
    write_profile = profile.copy()
    write_profile.update(dtype="float32", count=1, compress="lzw", nodata=float(FUEL_NODATA))
    values = np.asarray(data, dtype=np.float32)
    write_values = np.where(np.isfinite(values), values, float(FUEL_NODATA)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **write_profile) as dst:
        dst.write(write_values, 1)


def _write_intervention_rasters(
    *,
    baseline_fuel: np.ndarray,
    scenario_fuel: np.ndarray,
    raw_data_dir: Path,
    prediction_dir: Path,
    hex_id: str,
) -> None:
    paths = Paths(hex_id=hex_id, root_dir=raw_data_dir)
    reference_grid, profile = load_spatial_raster(
        path=paths.elevation_grid(hex_id),
        mask_path=paths.mask_grid_actual(hex_id),
    )
    height, width = reference_grid.shape
    if baseline_fuel.shape[0] < height or baseline_fuel.shape[1] < width:
        raise ValueError(f"Stitched fuel grid {baseline_fuel.shape} is smaller than prediction grid {(height, width)} for hex {hex_id}.")

    support = ~np.ma.getmaskarray(reference_grid)
    baseline = np.where(support, baseline_fuel[:height, :width], np.nan)
    scenario = np.where(support, scenario_fuel[:height, :width], np.nan)
    _write_fuel_raster(
        baseline,
        profile,
        fuel_intervention_raster_path(prediction_dir, hex_id, "baseline"),
    )
    _write_fuel_raster(
        scenario,
        profile,
        fuel_intervention_raster_path(prediction_dir, hex_id, "scenario"),
    )


class FuelCounterfactualTransform:
    """Dataset patch transform that overlays a per-hexel fuel-edit scenario.

    Instances are built once per scenario via `from_metadata`, which stitches each
    hexel's patches into a full fuel grid and applies the edit once. `__call__` slices
    the corresponding window from that edited grid and substitutes the patch's fuel
    channel at data-loading time.
    """

    def __init__(
        self,
        *,
        fuel_channel: int,
        filename_col: str,
        edited_hexels: dict[str, np.ndarray],
        patch_windows: dict[str, PatchFuelWindow],
        summary: pd.DataFrame,
        components: pd.DataFrame,
    ) -> None:
        self.fuel_channel = fuel_channel
        self.filename_col = filename_col
        self.edited_hexels = edited_hexels
        self.patch_windows = patch_windows
        self.summary = summary
        self.components = components

    @classmethod
    def from_metadata(
        cls,
        *,
        data_root: Path,
        metadata: pd.DataFrame,
        fuel_channel: int,
        scenario: ScenarioConfig,
        filename_col: str = "filename",
        prediction_dir: Path | None = None,
        raw_data_dir: Path | None = None,
    ) -> FuelCounterfactualTransform:
        """Precompute the edited fuel channel for every patch listed in `metadata`.

        For each hexel, patches are stitched into a single fuel grid (averaging
        overlapping windows) and the scenario's edit is applied once on that grid.
        When output paths are supplied, the exact baseline and edited grids are also
        persisted on the prediction raster grid for downstream analysis.
        """
        params = dict(scenario.fuel_edit() or {})
        mode = str(params.pop("mode", "nonfuel_to_burnable_local_adjacent_modal"))
        nonfuel_ids = params.pop("nonfuel_ids", None)
        if not isinstance(nonfuel_ids, list | tuple) or not nonfuel_ids:
            raise ValueError(f"Fuel scenario {scenario.name!r} must define nonfuel_ids.")
        if "hex_id" not in metadata.columns:
            raise ValueError("Patch metadata is missing required column 'hex_id'.")
        if (prediction_dir is None) != (raw_data_dir is None):
            raise ValueError("prediction_dir and raw_data_dir must be provided together.")

        normalized_hex_ids = metadata["hex_id"].astype(str).map(normalize_hex_id)
        edited_hexels: dict[str, np.ndarray] = {}
        patch_windows: dict[str, PatchFuelWindow] = {}
        summary_rows: list[dict[str, Any]] = []
        component_frames: list[pd.DataFrame] = []

        for hex_id in sorted(normalized_hex_ids.unique()):
            hex_metadata = metadata.loc[normalized_hex_ids == hex_id].drop_duplicates(filename_col)
            records = []
            windows = []
            masks = []
            coords = []
            max_row = 0
            max_col = 0
            for _, item in hex_metadata.iterrows():
                relative_path = Path(str(item[filename_col]))
                patch = np.load(data_root / relative_path, mmap_mode="r")
                fuel = np.asarray(patch[:, :, fuel_channel], dtype=np.float32)
                row = int(item["row"])
                col = int(item["col"])
                records.append((relative_path.as_posix(), row, col, fuel.shape))
                windows.append(fuel)
                masks.append(np.isfinite(fuel))
                coords.append((row, col))
                max_row = max(max_row, row + fuel.shape[0])
                max_col = max(max_col, col + fuel.shape[1])

            stitched = stitch_windows(windows, coords, masks, (max_row, max_col), mode="mean")
            result = apply_fuel_edit(
                stitched,
                [int(value) for value in nonfuel_ids],
                mode=mode,
                scenario_name=scenario.name,
                params=params,
            )
            edited_hexels[hex_id] = np.asarray(result.fuel, dtype=np.float32)
            for key, row, col, shape in records:
                if key in patch_windows:
                    raise ValueError(f"Duplicate patch filename in counterfactual metadata: {key}")
                patch_windows[key] = PatchFuelWindow(
                    hex_id=hex_id,
                    row=row,
                    col=col,
                    shape=shape,
                )

            if prediction_dir is not None and raw_data_dir is not None:
                _write_intervention_rasters(
                    baseline_fuel=stitched,
                    scenario_fuel=result.fuel,
                    raw_data_dir=raw_data_dir,
                    prediction_dir=prediction_dir,
                    hex_id=hex_id,
                )

            summary = asdict(result.report)
            summary["hex_id"] = hex_id
            summary_rows.append(summary)
            if not result.components.empty:
                components = result.components.copy()
                components.insert(0, "hex_id", hex_id)
                components.insert(0, "scenario_name", scenario.name)
                component_frames.append(components)

        return cls(
            fuel_channel=fuel_channel,
            filename_col=filename_col,
            edited_hexels=edited_hexels,
            patch_windows=patch_windows,
            summary=pd.DataFrame(summary_rows),
            components=pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame(),
        )

    def __call__(self, data: np.ndarray, patch_info: dict[str, Any]) -> np.ndarray:
        """Return a copy of `data` with its fuel channel replaced by the precomputed edit."""
        key = Path(str(patch_info[self.filename_col])).as_posix()
        patch_window = self.patch_windows.get(key)
        if patch_window is None:
            raise KeyError(f"No counterfactual fuel channel was prepared for patch {key}.")
        edited_hexel = self.edited_hexels[patch_window.hex_id]
        row_end = patch_window.row + patch_window.shape[0]
        col_end = patch_window.col + patch_window.shape[1]
        edited_channel = edited_hexel[patch_window.row : row_end, patch_window.col : col_end]
        if edited_channel.shape != patch_window.shape:
            raise ValueError(f"Edited fuel slice for patch {key} has shape {edited_channel.shape}; expected {patch_window.shape}.")
        edited = np.array(data, copy=True)
        edited[:, :, self.fuel_channel] = edited_channel.astype(edited.dtype, copy=False)
        return edited
