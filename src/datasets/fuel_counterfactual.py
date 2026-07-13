"""In-memory fuel counterfactual transform for prepared patch datasets."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.postprocessing.counterfactual import ScenarioConfig
from src.datasets.postprocessing.counterfactual_fuel import apply_fuel_edit
from src.datasets.postprocessing.stitch_hexel import stitch_windows


class FuelCounterfactualTransform:
    def __init__(
        self,
        *,
        fuel_channel: int,
        filename_col: str,
        edited_channels: dict[str, np.ndarray],
        summary: pd.DataFrame,
        components: pd.DataFrame,
    ) -> None:
        self.fuel_channel = fuel_channel
        self.filename_col = filename_col
        self.edited_channels = edited_channels
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
    ) -> FuelCounterfactualTransform:
        params = dict(scenario.fuel_edit() or {})
        mode = str(params.pop("mode", "nonfuel_to_burnable_local_adjacent_modal"))
        nonfuel_ids = params.pop("nonfuel_ids", None)
        if not isinstance(nonfuel_ids, list | tuple) or not nonfuel_ids:
            raise ValueError(f"Fuel scenario {scenario.name!r} must define nonfuel_ids.")
        if "hex_id" not in metadata.columns:
            raise ValueError("Patch metadata is missing required column 'hex_id'.")

        normalized_hex_ids = metadata["hex_id"].astype(str).str.extract(r"(\d+)", expand=False).str.zfill(2)
        edited_channels: dict[str, np.ndarray] = {}
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
            for key, row, col, shape in records:
                edited_channels[key] = result.fuel[row : row + shape[0], col : col + shape[1]].copy()

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
            edited_channels=edited_channels,
            summary=pd.DataFrame(summary_rows),
            components=pd.concat(component_frames, ignore_index=True) if component_frames else pd.DataFrame(),
        )

    def __call__(self, data: np.ndarray, patch_info: dict[str, Any]) -> np.ndarray:
        key = Path(str(patch_info[self.filename_col])).as_posix()
        if key not in self.edited_channels:
            raise KeyError(f"No counterfactual fuel channel was prepared for patch {key}.")
        edited = np.array(data, copy=True)
        edited[:, :, self.fuel_channel] = self.edited_channels[key].astype(edited.dtype, copy=False)
        return edited
