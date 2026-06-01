from pathlib import Path

import numpy as np

from src.config import GlobalContextParams
from src.datasets.sources.base import DataSource


class GlobalContextSource(DataSource):
    """Loads a downsampled full-hex context raster for each patch."""

    def __init__(
        self,
        root_dir: str,
        params: GlobalContextParams,
        modelling_approach: str = "1",
        transform=None,
    ):
        self.root_dir = root_dir
        self.params = params
        self.modelling_approach = modelling_approach
        self.transform = transform
        self.context_dir = Path(params.context_dir)
        if not self.context_dir.is_absolute():
            self.context_dir = Path(self.root_dir) / self.context_dir
        self.num_context_channels = params.num_context_channels
        self.context_filename_template = params.context_filename_template
        self.metadata_key_col = params.metadata_key_col
        self.include_patch_footprint = params.include_patch_footprint
        self.patch_height = params.patch_height
        self.patch_width = params.patch_width
        self._context_cache: dict[str, tuple[np.ndarray, int, int]] = {}

        if not self.context_dir.exists():
            raise FileNotFoundError(f"Global context directory not found: {self.context_dir}")

    @staticmethod
    def _normalize_key(value) -> str:
        if isinstance(value, str):
            stripped = value.strip()
            if stripped.isdigit():
                return str(int(stripped))
            return stripped
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        if isinstance(value, (float, np.floating)) and float(value).is_integer():
            return str(int(value))
        return str(value)

    def _context_path(self, key: str) -> Path:
        try:
            key_int: int | str = int(key)
        except ValueError:
            key_int = key
        filename = self.context_filename_template.format(
            hex_id=key,
            hex_id_int=key_int,
            hex_id_padded=str(key).zfill(2),
        )
        return self.context_dir / filename

    def _load_context(self, key: str) -> tuple[np.ndarray, int, int]:
        cached = self._context_cache.get(key)
        if cached is not None:
            return cached

        context_path = self._context_path(key)
        if not context_path.exists():
            raise FileNotFoundError(f"Global context file not found for hex_id={key!r}: {context_path}")

        with np.load(context_path) as data:
            context = data["context"].astype(np.float32)
            full_height = int(data["full_height"])
            full_width = int(data["full_width"])

        if context.ndim != 3:
            raise ValueError(f"Expected global context with shape (C,H,W), got {context.shape} in {context_path}")
        if context.shape[0] != self.num_context_channels:
            raise ValueError(f"Expected {self.num_context_channels} context channels in {context_path}, got {context.shape[0]}.")
        if not np.all(np.isfinite(context)):
            raise ValueError(f"Global context file contains non-finite values: {context_path}")
        result = (context, full_height, full_width)
        self._context_cache[key] = result
        return result

    @staticmethod
    def _patch_footprint(
        row: int,
        col: int,
        patch_height: int,
        patch_width: int,
        full_height: int,
        full_width: int,
        context_height: int,
        context_width: int,
    ) -> np.ndarray:
        footprint = np.zeros((context_height, context_width), dtype=np.float32)
        r0 = int(np.floor(row * context_height / full_height))
        r1 = int(np.ceil((row + patch_height) * context_height / full_height))
        c0 = int(np.floor(col * context_width / full_width))
        c1 = int(np.ceil((col + patch_width) * context_width / full_width))
        r0 = max(0, min(context_height, r0))
        r1 = max(r0 + 1, min(context_height, r1))
        c0 = max(0, min(context_width, c0))
        c1 = max(c0 + 1, min(context_width, c1))
        footprint[r0:r1, c0:c1] = 1.0
        return footprint

    def get_sample(self, patch_info: dict) -> np.ndarray:
        if self.metadata_key_col not in patch_info:
            raise KeyError(f"Patch metadata is missing required global-context key {self.metadata_key_col!r}.")

        key = self._normalize_key(patch_info[self.metadata_key_col])
        context, full_height, full_width = self._load_context(key)

        if not self.include_patch_footprint:
            return context

        if "row" not in patch_info or "col" not in patch_info:
            raise KeyError("Patch metadata must include 'row' and 'col' when include_patch_footprint=True.")

        footprint = self._patch_footprint(
            row=int(patch_info["row"]),
            col=int(patch_info["col"]),
            patch_height=self.patch_height,
            patch_width=self.patch_width,
            full_height=full_height,
            full_width=full_width,
            context_height=context.shape[1],
            context_width=context.shape[2],
        )
        return np.concatenate([context, footprint[np.newaxis, :, :]], axis=0)

    def input_dim(self):
        return self.num_context_channels + int(self.include_patch_footprint)
