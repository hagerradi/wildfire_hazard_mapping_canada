from pathlib import Path

import numpy as np
import pandas as pd

from src.datasets.fuel_counterfactual import FuelCounterfactualTransform
from src.datasets.postprocessing.counterfactual import ScenarioConfig


def test_fuel_counterfactual_reuses_stitching_for_overlapping_patches(tmp_path: Path) -> None:
    global_fuel = np.array(
        [
            [1, 1, 2, 2],
            [1, 0, 2, 2],
            [1, 1, 2, 2],
        ],
        dtype=np.float32,
    )
    patch_specs = [
        ("left.npy", 0, 0, global_fuel[:, :3]),
        ("right.npy", 0, 1, global_fuel[:, 1:]),
    ]
    for filename, _, _, fuel in patch_specs:
        patch = np.zeros((*fuel.shape, 2), dtype=np.float32)
        patch[:, :, 0] = fuel
        np.save(tmp_path / filename, patch)

    metadata = pd.DataFrame([{"filename": filename, "hex_id": 16, "row": row, "col": col} for filename, row, col, _ in patch_specs])
    scenario = ScenarioConfig(
        name="remove_barriers",
        kind="fuel",
        description="",
        params={
            "mode": "nonfuel_to_burnable_local_adjacent_modal",
            "nonfuel_groups": [0],
        },
    )

    transform = FuelCounterfactualTransform.from_metadata(
        data_root=tmp_path,
        metadata=metadata,
        fuel_channel=0,
        scenario=scenario,
    )

    left = transform(np.load(tmp_path / "left.npy"), metadata.iloc[0].to_dict())
    right = transform(np.load(tmp_path / "right.npy"), metadata.iloc[1].to_dict())
    assert left[1, 1, 0] == 1
    assert right[1, 0, 0] == 1
    assert transform.summary["edited_pixels"].tolist() == [1]
    assert transform.components["replacement_fuel_id"].tolist() == [1]
