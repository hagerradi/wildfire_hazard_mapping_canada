import os
from typing import Any

import numpy as np
import pandas as pd

from data_preparation.paths import Paths
from data_preparation.spatial.utils import fire_cause_mapping, load_spatial_raster


def load_ignition_grid(
    root_dir: str,
    hex_id: str,
    cause: int = None,
    season: int = None,
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str = "actual",
) -> np.ma.MaskedArray:
    """Load ignition grids for a specific season/cause or all seasons/causes"""
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ignition_grids_folder_path = all_paths.ignition_prob_dir()

    if season and cause and hex_id:
        file_name = f"hex{hex_id}_ignGrid_{fire_cause_mapping[cause]}_s{season}.tif"
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
            reference_profile=reference_profile,
        )

        return ignition_raster

    # else, loop over all ignition grids (across causes/seasons) and output one grid
    ignition_raster_files = [f for f in os.listdir(ignition_grids_folder_path) if f.endswith(".tif")]
    out_ignition_grids = []
    for file_name in ignition_raster_files:
        ignition_raster, _ = load_spatial_raster(
            path=ignition_grids_folder_path / file_name,
            mask_path=all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope),
            reference_profile=reference_profile,
        )
        out_ignition_grids.append(ignition_raster)

    # size (H, W)
    out_ignition_grids = np.ma.stack(out_ignition_grids, axis=0)  # type: ignore
    max_ignition_grid = np.ma.max(out_ignition_grids, axis=0)
    return max_ignition_grid


# Maps (cause_letter, season_int) → (grid_key, CSV cause label, season label)
_CAUSE_SEASON_SPECS = [
    ("H", 1, "Human", "s1"),
    ("H", 2, "Human", "s2"),
    ("N", 1, "Lightning", "s1"),
    ("N", 2, "Lightning", "s2"),
]


def load_ignition_grid_weighted(
    root_dir: str,
    hex_id: str,
    firezones_grid: np.ma.MaskedArray,
    reference_profile: dict[str, Any] | None = None,
    mask_scope: str = "actual",
) -> np.ma.MaskedArray:
    """Load ignition grids as two zone-area-weighted channels: Human and Lightning.

    For each hex, the IgnitionDistribution.csv gives a RelativeLikelihood per
    (Season, Cause, FireZone).  We compute a hex-level weight for each of the
    4 (cause × season) combinations by averaging RelativeLikelihoods across
    fire zones using their pixel-area fractions, then normalise all four weights
    to sum to 1.  The four grids are then blended into two output channels:

        human_channel     = w_H_s1 * grid_H_s1 + w_H_s2 * grid_H_s2
        lightning_channel = w_N_s1 * grid_N_s1 + w_N_s2 * grid_N_s2

    Returns a masked array of shape (H, W, 2) where channel 0 = Human,
    channel 1 = Lightning.  Falls back to uniform weights (equal weight per
    present TIF) if the CSV is missing or all weights are zero.
    """
    all_paths = Paths(hex_id=hex_id, root_dir=root_dir)
    ign_dir = all_paths.ignition_prob_dir()
    mask_path = all_paths.mask_grid(hex_id=hex_id, mask_scope=mask_scope)

    # ── 1. Load the four ignition TIFs by explicit name ───────────────────────
    # Some hexels may be missing certain cause/season TIFs (e.g. hex54 has no
    # lightning grids). Missing TIFs are skipped; their weight is forced to 0.
    grids: dict[tuple[str, int], np.ma.MaskedArray] = {}
    missing_keys: set[tuple[str, int]] = set()
    for cause_letter, season_int, _cause_label, _season_label in _CAUSE_SEASON_SPECS:
        fname = f"hex{hex_id}_ignGrid_{cause_letter}_s{season_int}.tif"
        tif_path = ign_dir / fname
        if not tif_path.exists():
            missing_keys.add((cause_letter, season_int))
            continue
        raster, _ = load_spatial_raster(
            path=tif_path,
            mask_path=mask_path,
            reference_profile=reference_profile,
        )
        grids[(cause_letter, season_int)] = raster

    if not grids:
        raise FileNotFoundError(f"No ignition TIFs found in {ign_dir} for hex{hex_id}")

    # ── 2. Compute area fractions over valid, non-zero zone pixels ─────────────
    zone_data = firezones_grid.data if isinstance(firezones_grid, np.ma.MaskedArray) else firezones_grid
    zone_mask = (
        np.ma.getmaskarray(firezones_grid) if isinstance(firezones_grid, np.ma.MaskedArray) else np.zeros_like(zone_data, dtype=bool)
    )
    valid_zone_pixels = (~zone_mask) & (zone_data > 0)
    zone_ids_valid = zone_data[valid_zone_pixels]

    area_frac: dict[int, float] = {}
    total_valid = zone_ids_valid.size
    if total_valid > 0:
        unique_ids, counts = np.unique(zone_ids_valid, return_counts=True)
        for zid, cnt in zip(unique_ids, counts):
            area_frac[int(zid)] = cnt / total_valid

    # ── 3. Load zone name → ID mapping from FireZones.csv ────────────────────
    fz_csv = all_paths.firezones_table(hex_id=hex_id)
    zone_name_to_id: dict[str, int] = {}
    if fz_csv.exists():
        fz_df = pd.read_csv(fz_csv)
        # Expected columns: Name, ID  (may have Description, Color)
        for _, row in fz_df.iterrows():
            name = str(row.get("Name", row.get("name", ""))).strip()
            zone_id = row.get("ID", row.get("id", None))
            if name and zone_id is not None:
                zone_name_to_id[name] = int(zone_id)

    # ── 4. Compute hex-level weights from IgnitionDistribution.csv ───────────
    ign_csv = all_paths.ignition_distribution_table(hex_id=hex_id)
    weights: dict[tuple[str, int], float] = {(c, s): 0.0 for c, s, _, _ in _CAUSE_SEASON_SPECS}

    if ign_csv.exists() and zone_name_to_id and area_frac:
        dist_df = pd.read_csv(ign_csv)
        # Normalise column names
        dist_df.columns = [col.strip() for col in dist_df.columns]
        for _, row in dist_df.iterrows():
            cause_csv = str(row.get("Cause", "")).strip()  # "Human" or "Lightning"
            season_csv = str(row.get("Season", "")).strip()  # "s1" or "s2"
            zone_name = str(row.get("FireZone", "")).strip()  # "fru21" etc.
            try:
                rl = float(row.get("RelativeLikelihood", 0) or 0)
            except (ValueError, TypeError):
                continue

            zone_id = zone_name_to_id.get(zone_name)
            if zone_id is None or zone_id not in area_frac:
                continue

            # Map CSV cause/season back to our (cause_letter, season_int) key
            cause_letter = "H" if cause_csv == "Human" else "N" if cause_csv == "Lightning" else None
            season_int = 1 if season_csv == "s1" else 2 if season_csv == "s2" else None
            if cause_letter is None or season_int is None:
                continue

            weights[(cause_letter, season_int)] += area_frac[zone_id] * rl
    else:
        # Fallback: uniform weights across present grids only
        for key in grids:
            weights[key] = 1.0

    # ── 5. Force missing TIFs to zero weight, then normalise ──────────────────
    for key in missing_keys:
        weights[key] = 0.0

    present_keys = set(grids.keys())
    total_w = sum(weights[k] for k in present_keys)
    if total_w > 0:
        weights = {k: (weights[k] / total_w if k in present_keys else 0.0) for k in weights}
    else:
        # All weights are zero (e.g. CSV has no matching zones); uniform over present
        n_present = len(present_keys)
        weights = {k: (1.0 / n_present if k in present_keys else 0.0) for k in weights}

    # ── 6. Blend into two output channels ────────────────────────────────────
    ref = next(iter(grids.values()))

    def _blend(keys: list[tuple[str, int]]) -> np.ma.MaskedArray:
        result = np.ma.zeros_like(ref)
        for k in keys:
            if k in grids:
                result = result + weights[k] * grids[k]
        return result

    human_channel = _blend([("H", 1), ("H", 2)])
    lightning_channel = _blend([("N", 1), ("N", 2)])

    # Stack to (H, W, 2)
    H, W = human_channel.shape
    out = np.ma.stack([human_channel, lightning_channel], axis=-1)  # (H, W, 2)
    assert out.shape == (H, W, 2)
    return out
