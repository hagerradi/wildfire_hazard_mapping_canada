from __future__ import annotations

import numpy as np
import pytest

from src.datasets.postprocessing.counterfactual_fuel import (
    burnable_mask,
    modal_adjacent_burnable_fuel,
    modal_adjacent_burnable_fuel_across_grids,
    nonfuel_mask,
    replace_burnable_with_nonfuel,
    replace_nonfuel_components_with_adjacent_modal,
    replace_nonfuel_with_adjacent_modal,
    replace_nonfuel_with_burnable,
)
from src.datasets.postprocessing.counterfactual_fuel_intervention_map import (
    intervention_layers,
    summarize_intervention,
)
from src.datasets.postprocessing.counterfactual_viz import zone_boundary_segments


def test_nonfuel_and_burnable_masks_are_complements_on_valid_fuel() -> None:
    fuel = np.array([[101, 1, -32768], [2, 100, 3]], dtype=np.int32)
    nonfuel = nonfuel_mask(fuel, [100, 101])
    burnable = burnable_mask(fuel, [100, 101])
    assert nonfuel.tolist() == [[True, False, False], [False, True, False]]
    assert burnable.tolist() == [[False, True, False], [True, False, True]]


def test_modal_adjacent_burnable_fuel_uses_local_neighbours() -> None:
    fuel = np.array(
        [
            [1, 1, 2],
            [1, 101, 2],
            [3, 3, 2],
        ],
        dtype=np.int32,
    )
    edit = fuel == 101
    burnable = fuel != 101
    replacement, n_candidates, note = modal_adjacent_burnable_fuel(fuel, edit, burnable)
    assert replacement == 1
    assert n_candidates == 8
    assert "adjacent" in note


def test_replace_nonfuel_with_adjacent_modal_reports_edit() -> None:
    fuel = np.array(
        [
            [1, 1, 2],
            [1, 101, 2],
            [3, 3, 2],
        ],
        dtype=np.int32,
    )
    edited, edit_mask, report = replace_nonfuel_with_adjacent_modal(fuel, [101])
    assert edit_mask.sum() == 1
    assert edited[1, 1] == 1
    assert report.edited_pixels == 1
    assert report.replacement_fuel_id == 1


def test_replace_nonfuel_components_with_adjacent_modal_uses_local_components() -> None:
    fuel = np.array(
        [
            [1, 1, 1, 2, 2, 2],
            [1, 0, 1, 2, 0, 2],
            [1, 1, 1, 2, 2, 2],
        ],
        dtype=np.float32,
    )
    edited, edit_mask, report, components = replace_nonfuel_components_with_adjacent_modal(fuel, [0])
    assert edit_mask.sum() == 2
    assert edited[1, 1] == pytest.approx(1.0)
    assert edited[1, 4] == pytest.approx(2.0)
    assert components["replacement_fuel_id"].tolist() == [1, 2]
    assert report.edited_pixels == 2
    assert "2 connected components" in report.note


def test_intervention_layers_show_only_replaced_nonfuel_pixels() -> None:
    baseline = np.array(
        [
            [1, 0, 2],
            [3, 0, np.nan],
        ],
        dtype=np.float32,
    )
    scenario = np.array(
        [
            [1, 8, 2],
            [3, 14, np.nan],
        ],
        dtype=np.float32,
    )

    original_nonfuel, replacement_map, unexpected = intervention_layers(baseline, scenario)
    assert original_nonfuel.tolist() == [[False, True, False], [False, True, False]]
    assert np.isnan(replacement_map[0, 0])
    assert replacement_map[0, 1] == pytest.approx(8)
    assert replacement_map[1, 1] == pytest.approx(14)
    assert not unexpected.any()

    summary = summarize_intervention(
        scenario="synthetic",
        endpoint="bp",
        hex_id="16",
        baseline_fuel=baseline,
        replacement_map=replacement_map,
        original_nonfuel=original_nonfuel,
        unexpected_burnable_changes=unexpected,
    )
    assert summary.n_original_nonfuel_pixels == 2
    assert summary.n_replaced_pixels == 2
    assert summary.replacement_fuel_ids == "8;14"


def test_modal_adjacent_burnable_across_grids_picks_consistent_replacement() -> None:
    grids = [
        np.array([[1, 101, 2], [1, 101, 2]], dtype=np.int32),
        np.array([[2, 101, 2], [3, 3, 2]], dtype=np.int32),
    ]
    replacement, candidate_pixels, note = modal_adjacent_burnable_fuel_across_grids(grids, [101])
    assert replacement == 2
    assert candidate_pixels > 0
    assert "across grids" in note


def test_replace_nonfuel_with_burnable_uses_fixed_replacement() -> None:
    fuel = np.array([[1, 101, 2], [100, 2, -32768]], dtype=np.int32)
    edited, edit_mask, report = replace_nonfuel_with_burnable(fuel, [100, 101], 2)
    assert edit_mask.tolist() == [[False, True, False], [True, False, False]]
    assert edited.tolist() == [[1, 2, 2], [2, 2, -32768]]
    assert report.edited_pixels == 2
    assert report.replacement_fuel_id == 2


def test_fuel_edits_preserve_nan_support_pixels() -> None:
    fuel = np.array([[1.0, 101.0, np.nan], [100.0, 2.0, np.nan]], dtype=np.float32)
    edited, edit_mask, report = replace_nonfuel_with_burnable(fuel, [100, 101], 2)
    assert edit_mask.tolist() == [[False, True, False], [True, False, False]]
    assert np.isnan(edited[0, 2])
    assert np.isnan(edited[1, 2])
    assert edited[0, 1] == pytest.approx(2.0)
    assert edited[1, 0] == pytest.approx(2.0)
    assert report.original_burnable_pixels == 2


def test_replace_burnable_with_nonfuel_only_edits_burnable_pixels() -> None:
    fuel = np.array(
        [
            [1, 101, 2],
            [1, 2, -32768],
        ],
        dtype=np.int32,
    )
    insertion = np.ones_like(fuel, dtype=bool)
    edited, edit_mask, report = replace_burnable_with_nonfuel(fuel, [101], insertion, replacement_nonfuel_id=101)
    assert edit_mask.tolist() == [[True, False, True], [True, True, False]]
    assert np.all(edited[edit_mask] == 101)
    assert edited[0, 1] == 101
    assert edited[1, 2] == -32768
    assert report.edited_pixels == 4


def test_zone_boundary_segments_traces_only_valid_interzone_borders() -> None:
    labels = np.ma.masked_array(
        np.array([[1, 1, 2], [1, 3, 2], [1, 3, 2]], dtype=np.int64),
        mask=np.zeros((3, 3), dtype=bool),
    )
    segments = zone_boundary_segments(labels)
    edges = {(tuple(np.round(a, 3)), tuple(np.round(b, 3))) for a, b in segments}
    expected = {
        ((0.5, 0.5), (0.5, 1.5)),
        ((0.5, 1.5), (0.5, 2.5)),
        ((1.5, -0.5), (1.5, 0.5)),
        ((1.5, 0.5), (1.5, 1.5)),
        ((1.5, 1.5), (1.5, 2.5)),
        ((0.5, 0.5), (1.5, 0.5)),
    }
    assert edges == expected


def test_zone_boundary_segments_skips_masked_neighbours() -> None:
    labels = np.ma.masked_array(
        np.array([[1, 2], [1, 2]], dtype=np.int64),
        mask=np.array([[False, True], [False, False]], dtype=bool),
    )
    segments = zone_boundary_segments(labels)
    assert segments.shape[0] == 1
    (start, end) = segments[0]
    assert tuple(np.round(start, 3)) == (0.5, 0.5)
    assert tuple(np.round(end, 3)) == (0.5, 1.5)
