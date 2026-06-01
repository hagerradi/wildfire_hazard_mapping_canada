from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import torch
from rasterio.windows import Window

from data_preparation.paths import Paths, prepared_mask_scope
from data_preparation.spatial.utils import load_spatial_raster
from src.config import BPPredictionParams
from src.datasets.sources.base import DataSource


class BPPredictionSource(DataSource):
    """Append a precomputed BP prediction raster as a patch-aligned scalar channel."""

    def __init__(
        self,
        root_dir: str,
        params: BPPredictionParams,
        modelling_approach: str = "1",
        transform=None,
        raw_data_dir: str | None = None,
    ):
        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform
        self.raw_data_dir = raw_data_dir

        self.prediction_dir = Path(params.prediction_dir)
        if not self.prediction_dir.is_absolute():
            self.prediction_dir = Path(self.root_dir) / self.prediction_dir
        self.filename_template = params.filename_template
        self.metadata_key_col = params.metadata_key_col
        self.fill_value = params.fill_value
        self.clip_min = params.clip_min
        self.clip_max = params.clip_max
        self.validate_alignment = params.validate_alignment
        self._profile_cache: dict[tuple[str, str], dict[str, Any]] = {}

        if not self.prediction_dir.exists():
            raise FileNotFoundError(f"BP prediction directory not found: {self.prediction_dir}")
        if self.validate_alignment and not self.raw_data_dir:
            raise ValueError("BPPredictionSource requires raw_data_dir when validate_alignment=True.")

    @staticmethod
    def _normalize_hex_id(value: object) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return str(int(stripped)).zfill(2)
            return stripped
        if isinstance(value, (int, np.integer)):
            return str(int(value)).zfill(2)
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value)).zfill(2)
        return str(value)

    @staticmethod
    def _mask_scope(patch_info: dict) -> str:
        return prepared_mask_scope(str(patch_info.get("mask_scope", "actual")))

    def _prediction_path(self, hex_id: str) -> Path:
        try:
            hex_id_int: int | str = int(hex_id)
        except ValueError:
            hex_id_int = hex_id
        filename = self.filename_template.format(
            hex_id=hex_id,
            hex_id_int=hex_id_int,
            hex_id_padded=str(hex_id).zfill(2),
        )
        return self.prediction_dir / filename

    def _reference_profile(self, hex_id: str, mask_scope: str) -> dict[str, Any]:
        cache_key = (hex_id, mask_scope)
        cached = self._profile_cache.get(cache_key)
        if cached is not None:
            return cached

        if not self.raw_data_dir:
            raise ValueError("raw_data_dir is required to validate BP prediction alignment.")
        paths = Paths(hex_id=hex_id, root_dir=self.raw_data_dir)
        _, profile = load_spatial_raster(
            path=paths.elevation_grid(hex_id=hex_id),
            mask_path=paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
        )
        self._profile_cache[cache_key] = profile
        return profile

    @staticmethod
    def _assert_aligned(pred_path: Path, pred_profile: dict[str, Any], ref_profile: dict[str, Any]) -> None:
        pred_shape = (int(pred_profile["height"]), int(pred_profile["width"]))
        ref_shape = (int(ref_profile["height"]), int(ref_profile["width"]))
        if pred_shape != ref_shape:
            raise ValueError(f"BP prediction grid shape {pred_shape} does not match reference grid {ref_shape}: {pred_path}")
        if pred_profile.get("crs") != ref_profile.get("crs"):
            raise ValueError(f"BP prediction CRS does not match reference grid: {pred_path}")
        if pred_profile.get("transform") != ref_profile.get("transform"):
            raise ValueError(f"BP prediction transform does not match reference grid: {pred_path}")

    def get_sample(self, patch_info: dict):
        if self.metadata_key_col not in patch_info:
            raise KeyError(f"Patch metadata is missing required BP prediction key {self.metadata_key_col!r}.")
        if "row" not in patch_info or "col" not in patch_info:
            raise KeyError("Patch metadata must include 'row' and 'col' to crop BP predictions.")
        if "data" not in patch_info:
            raise KeyError("Patch data must be loaded before cropping BP predictions.")

        data = patch_info["data"]
        patch_height, patch_width = data.shape[:2]
        row = int(patch_info["row"])
        col = int(patch_info["col"])
        hex_id = self._normalize_hex_id(patch_info[self.metadata_key_col])
        mask_scope = self._mask_scope(patch_info)
        pred_path = self._prediction_path(hex_id)
        if not pred_path.exists():
            raise FileNotFoundError(f"BP prediction raster not found for hex_id={hex_id!r}: {pred_path}")

        with rasterio.open(pred_path) as src:
            if self.validate_alignment:
                self._assert_aligned(pred_path, src.profile, self._reference_profile(hex_id, mask_scope))
            if row < 0 or col < 0 or row >= src.height or col >= src.width:
                raise ValueError(
                    f"Patch window row={row}, col={col}, shape={(patch_height, patch_width)} does not overlap BP prediction "
                    f"shape={(src.height, src.width)} for {pred_path}"
                )
            window = Window(col_off=col, row_off=row, width=patch_width, height=patch_height)
            bp = src.read(1, window=window, masked=True, boundless=True, fill_value=self.fill_value)

        channel = np.asarray(np.ma.filled(bp, self.fill_value), dtype=np.float32)
        if self.clip_min is not None or self.clip_max is not None:
            channel = np.clip(
                channel,
                self.clip_min if self.clip_min is not None else -np.inf,
                self.clip_max if self.clip_max is not None else np.inf,
            )
        if not np.all(np.isfinite(channel)):
            raise ValueError(f"BP prediction channel contains non-finite values after nodata fill: {pred_path}")
        return torch.from_numpy(channel[np.newaxis, :, :])

    def input_dim(self):
        return 1
