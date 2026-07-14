from src.datasets.postprocessing.counterfactual import ScenarioConfig
from src.evaluate_counterfactual import _select_scenarios

_BASELINE = ScenarioConfig(name="baseline", kind="baseline", description="", params={})
_FUEL_A = ScenarioConfig(name="fuel_a", kind="fuel", description="", params={})
_FUEL_B = ScenarioConfig(name="fuel_b", kind="fuel", description="", params={})
_ALL_SCENARIOS = [_BASELINE, _FUEL_A, _FUEL_B]


def test_select_scenarios_returns_all_when_unfiltered() -> None:
    assert _select_scenarios(_ALL_SCENARIOS, None) == _ALL_SCENARIOS


def test_select_scenarios_always_keeps_baseline() -> None:
    selected = _select_scenarios(_ALL_SCENARIOS, {"fuel_a"})
    assert selected == [_BASELINE, _FUEL_A]
