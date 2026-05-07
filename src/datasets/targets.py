from dataclasses import dataclass
from typing import Literal, cast

TargetName = Literal["bp", "fi", "ros"]


@dataclass(frozen=True)
class TargetSpec:
    name: TargetName
    channel_key: str
    output_type: str
    path_method: str
    label: str
    probability_scale: bool = False


TARGET_SPECS: dict[TargetName, TargetSpec] = {
    "bp": TargetSpec(
        name="bp",
        channel_key="bp_out_grid",
        output_type="fire_burn_probability",
        path_method="output_burn_prob",
        label="Burn Probability",
        probability_scale=True,
    ),
    "fi": TargetSpec(
        name="fi",
        channel_key="fi_out_grid",
        output_type="fire_intensity",
        path_method="output_fire_intensity",
        label="Fire Intensity",
    ),
    "ros": TargetSpec(
        name="ros",
        channel_key="ros_out_grid",
        output_type="fire_ros",
        path_method="output_ros",
        label="Rate of Spread",
    ),
}

TARGET_ALIASES = {
    "burn_probability": "bp",
    "fire_burn_probability": "bp",
    "bp_out_grid": "bp",
    "fire_intensity": "fi",
    "fi_out_grid": "fi",
    "rate_of_spread": "ros",
    "fire_ros": "ros",
    "ros_out_grid": "ros",
}


def get_target_spec(target_name: str) -> TargetSpec:
    normalized = target_name.strip().lower()
    normalized = TARGET_ALIASES.get(normalized, normalized)
    if normalized not in TARGET_SPECS:
        supported = sorted(set(TARGET_SPECS) | set(TARGET_ALIASES))
        raise ValueError(f"Unsupported target_name={target_name!r}. Supported values: {supported}")
    return TARGET_SPECS[cast(TargetName, normalized)]
