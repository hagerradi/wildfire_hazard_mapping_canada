from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.datasets.postprocessing.counterfactual.counterfactual_base import EndpointConfig, ScenarioConfig
from src.evaluate_counterfactual import _select_endpoints, _select_scenarios, run_counterfactual_evaluation

_BASELINE = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
_FUEL_A = ScenarioConfig(name="fuel_a", kind="fuel", description="", params={})
_FUEL_B = ScenarioConfig(name="fuel_b", kind="fuel", description="", params={})
_ALL_SCENARIOS = [_BASELINE, _FUEL_A, _FUEL_B]


def test_select_scenarios_returns_all_when_unfiltered() -> None:
    assert _select_scenarios(_ALL_SCENARIOS, None) == _ALL_SCENARIOS


def test_select_scenarios_always_keeps_baseline() -> None:
    selected = _select_scenarios(_ALL_SCENARIOS, {"fuel_a"})
    assert selected == [_BASELINE, _FUEL_A]


def test_select_scenarios_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match=r"Unknown scenario selection.*missing"):
        _select_scenarios(_ALL_SCENARIOS, {"missing"})


def test_select_endpoints_rejects_unknown_names() -> None:
    endpoints = {"bp": EndpointConfig(name="bp", config_path=Path("bp.yaml"))}
    with pytest.raises(ValueError, match=r"Unknown endpoint selection.*fi"):
        _select_endpoints(endpoints, {"fi"})


@dataclass
class _DataConfig:
    root_dir: str
    test_split: str = "test.csv"
    valid_mask_threshold: float = 0.5
    filename_col: str = "filename"
    num_workers: int = 4
    raw_data_dir: str = ""


@dataclass
class _EvaluationConfig:
    checkpoint_filename: str = "best.pt"


@dataclass
class _LoggerConfig:
    enabled: bool = True


@dataclass
class _EndpointRunConfig:
    save_dir: str
    data: _DataConfig
    evaluation: _EvaluationConfig = field(default_factory=_EvaluationConfig)
    logger: _LoggerConfig = field(default_factory=_LoggerConfig)
    modelling_approach: str = "common_input_pipeline"

    def model_copy(self, *, deep: bool) -> _EndpointRunConfig:
        return deepcopy(self) if deep else self


class _FakeFuelTransform:
    summary = pd.DataFrame([{"scenario_name": "fuel_a", "edited_pixels": 12}])
    components = pd.DataFrame()
    calls: list[dict] = []

    @classmethod
    def from_metadata(cls, **kwargs):
        cls.calls.append(kwargs)
        return cls()


def test_run_counterfactual_evaluation_orchestrates_selected_scenario(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root = tmp_path
    data_root = project_root / "data"
    source_dir = project_root / "trained"
    save_dir = project_root / "counterfactual"
    data_root.mkdir()
    source_dir.mkdir()
    (source_dir / "best.pt").write_bytes(b"checkpoint")
    pd.DataFrame(
        [
            {
                "hex_id": "16",
                "filename": "patch.npy",
                "valid_ratio": 1.0,
            }
        ]
    ).to_csv(data_root / "test.csv", index=False)
    config_path = project_root / "counterfactual.yaml"
    with config_path.open("w") as handle:
        yaml.safe_dump(
            {
                "raw_data_dir": "raw",
                "save_dir": "counterfactual",
                "hex_ids": ["16"],
                "endpoints": {"bp": {"config_path": "bp.yaml"}},
                "scenarios": [
                    {"name": "baseline", "kind": "baseline"},
                    {"name": "fuel_a", "kind": "fuel"},
                ],
            },
            handle,
        )

    run_config = _EndpointRunConfig(
        save_dir=str(source_dir),
        data=_DataConfig(root_dir=str(data_root)),
    )
    evaluation_calls: list[dict] = []

    def _fake_evaluate_hexels(**kwargs):
        evaluation_calls.append(kwargs)
        return {"mae": 1.5}

    monkeypatch.setattr("src.evaluate_counterfactual.load_config", lambda _: run_config)
    monkeypatch.setattr("src.evaluate_counterfactual._fuel_channel", lambda *_: 3)
    monkeypatch.setattr("src.evaluate_counterfactual.FuelCounterfactualTransform", _FakeFuelTransform)
    monkeypatch.setattr("src.evaluate_counterfactual.evaluate_hexels", _fake_evaluate_hexels)
    _FakeFuelTransform.calls.clear()
    stale_components = save_dir / "fuel_component_replacements.csv"
    save_dir.mkdir()
    stale_components.write_text("stale\n")

    index = run_counterfactual_evaluation(
        config_path,
        endpoint_names={"bp"},
        scenario_names={"fuel_a"},
        overwrite=False,
        project_root=project_root,
    )

    assert index[["scenario", "endpoint"]].to_dict("records") == [
        {"scenario": "baseline", "endpoint": "bp"},
        {"scenario": "fuel_a", "endpoint": "bp"},
    ]
    assert len(evaluation_calls) == 2
    assert evaluation_calls[0]["patch_transform"] is None
    assert isinstance(evaluation_calls[1]["patch_transform"], _FakeFuelTransform)
    assert len(_FakeFuelTransform.calls) == 1
    assert not stale_components.exists()
    assert (save_dir / "scenario_prediction_index.csv").exists()
    assert (save_dir / "counterfactual_metrics.csv").exists()
    summary = pd.read_csv(save_dir / "fuel_edit_summary.csv")
    assert summary[["endpoint", "edited_pixels"]].to_dict("records") == [{"endpoint": "bp", "edited_pixels": 12}]
