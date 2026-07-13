from pathlib import Path

import pytest
import yaml

from src.datasets.postprocessing.counterfactual import load_counterfactual_config


def _write_config(path: Path, data: dict) -> None:
    with path.open("w") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)


def test_load_counterfactual_config_supports_custom_endpoints(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["hex1", 16],
            "endpoints": {
                "hazard": {
                    "config_path": "configs/hazard.yaml",
                }
            },
            "scenarios": [
                {"name": "baseline", "kind": "baseline"},
                {
                    "name": "remove_barriers",
                    "kind": "fuel",
                    "params": {"mode": "nonfuel_to_burnable_local_adjacent_modal"},
                },
            ],
        },
    )

    config = load_counterfactual_config(path)

    assert config.hex_ids == ["01", "16"]
    assert list(config.endpoints) == ["hazard"]
    assert config.scenarios[1].fuel_edit() == {"mode": "nonfuel_to_burnable_local_adjacent_modal"}


def test_load_counterfactual_config_requires_baseline(tmp_path: Path) -> None:
    path = tmp_path / "counterfactual.yaml"
    _write_config(
        path,
        {
            "raw_data_dir": "/raw",
            "save_dir": "/experiment",
            "hex_ids": ["16"],
            "endpoints": {"bp": {"config_path": "bp.yaml"}},
            "scenarios": [{"name": "edit", "kind": "fuel"}],
        },
    )

    with pytest.raises(ValueError, match="baseline"):
        load_counterfactual_config(path)
