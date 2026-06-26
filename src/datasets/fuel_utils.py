from pathlib import Path

import numpy as np
import pandas as pd

from data_preparation.paths import Paths
from data_preparation.utils import find_hex_ids


def _normalize_hex_id(hex_id: str | int) -> str:
    """
    Normalize hex IDs to two digits.

    Examples:
        1    -> "01"
        "1"  -> "01"
        "01" -> "01"
    """
    return str(hex_id).replace("hex", "").zfill(2)


def read_ros_curves(
    ros_csv_path: str | Path,
    code_col: str = "fbp_code",
    season_col: str = "SeasonState",
    isi_col: str = "ISI",
    ros_col: str = "ROS",
) -> dict[int, dict[str, pd.Series]]:
    """
    Read the ROS curve CSV once.

    Returns
    -------
    dict
        {
            fbp_code: {
                SeasonState: pd.Series(
                    index=ISI,
                    values=ROS,
                )
            }
        }
    """
    ros_csv_path = Path(ros_csv_path)

    if not ros_csv_path.exists():
        raise FileNotFoundError(f"ROS curve CSV not found: {ros_csv_path}")

    df = pd.read_csv(ros_csv_path)

    required_cols = {
        code_col,
        season_col,
        isi_col,
        ros_col,
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError("Missing required columns in ROS CSV: " f"{sorted(missing_cols)}")

    df = df[
        [
            code_col,
            season_col,
            isi_col,
            ros_col,
        ]
    ].copy()

    df[code_col] = pd.to_numeric(
        df[code_col],
        errors="raise",
    ).astype(int)

    df[isi_col] = pd.to_numeric(
        df[isi_col],
        errors="raise",
    )

    df[ros_col] = pd.to_numeric(
        df[ros_col],
        errors="raise",
    )

    df[season_col] = df[season_col].astype(str).str.strip()

    duplicate_mask = df.duplicated(
        subset=[
            code_col,
            season_col,
            isi_col,
        ],
        keep=False,
    )

    if duplicate_mask.any():
        duplicates = (
            df.loc[
                duplicate_mask,
                [
                    code_col,
                    season_col,
                    isi_col,
                ],
            ]
            .drop_duplicates()
            .sort_values(
                [
                    code_col,
                    season_col,
                    isi_col,
                ]
            )
        )

        raise ValueError(
            "Multiple ROS values were found for the same " "fbp_code, SeasonState, and ISI:\n" f"{duplicates.to_string(index=False)}"
        )

    curves_by_code: dict[int, dict[str, pd.Series]] = {}

    for fbp_code, code_group in df.groupby(
        code_col,
        sort=False,
    ):
        season_curves: dict[str, pd.Series] = {}

        for season_state, season_group in code_group.groupby(
            season_col,
            sort=False,
        ):
            curve = season_group.sort_values(isi_col).set_index(isi_col)[ros_col].astype(float)

            season_curves[str(season_state)] = curve

        curves_by_code[int(fbp_code)] = season_curves

    return curves_by_code


def _read_hex_season_weights(
    distribution_path: str | Path,
    season_mapping: dict[str, str],
) -> dict[str, float]:
    """
    Read one hex ignition distribution file and calculate one
    normalized weight per ROS SeasonState.

    Example:

        ,,s1,Lightning,fru21,0.5
        ,,s2,Lightning,fru21,1.5

    Parameters
    ----------
    season_mapping
        Maps ignition seasons to ROS SeasonState values.

        Example:
            {
                "s1": "leafless",
                "s2": "green",
            }

    Returns
    -------
    dict
        Example:

            {
                "leafless": 0.09,
                "green": 0.91,
            }
    """
    distribution_path = Path(distribution_path)

    if not distribution_path.exists():
        raise FileNotFoundError("Ignition distribution CSV not found: " f"{distribution_path}")

    distribution_df = pd.read_csv(distribution_path)

    distribution_df["Season"] = distribution_df["Season"].astype(str).str.strip()

    distribution_df["RelativeLikelihood"] = pd.to_numeric(
        distribution_df["RelativeLikelihood"],
        errors="raise",
    )

    ignition_weights = distribution_df.groupby("Season")["RelativeLikelihood"].sum().to_dict()

    season_state_weights: dict[str, float] = {}

    for ignition_season, season_state in season_mapping.items():
        weight = float(
            ignition_weights.get(
                ignition_season,
                0.0,
            )
        )

        season_state_weights[season_state] = (
            season_state_weights.get(
                season_state,
                0.0,
            )
            + weight
        )

    total_weight = sum(season_state_weights.values())

    if total_weight <= 0:
        raise ValueError("The ignition distribution contains no positive " f"season weight: {distribution_path}")

    return {season_state: weight / total_weight for season_state, weight in season_state_weights.items()}


def _combine_season_curves(
    fbp_code: int,
    hex_id: str,
    season_curves: dict[str, pd.Series],
    season_weights: dict[str, float],
) -> np.ndarray:
    """
    Return one ROS vector for one fbp_code and one hex_id.

    If only one SeasonState exists, that curve is returned directly.

    If multiple SeasonState curves exist, they are aligned by ISI and
    combined using the hex-specific ignition season weights.
    """
    season_states = list(season_curves.keys())

    if not season_states:
        raise ValueError(f"No ROS curves found for fbp_code={fbp_code}")

    # Only one SeasonState exists for this code.
    if len(season_states) == 1:
        only_state = season_states[0]

        return season_curves[only_state].sort_index().to_numpy(dtype=np.float32)

    missing_weights = [season_state for season_state in season_states if season_state not in season_weights]

    if missing_weights:
        raise ValueError(
            "No ignition season weight was found for "
            f"fbp_code={fbp_code}, hex_id={hex_id}, "
            f"SeasonState={missing_weights}. "
            f"Available weights={season_weights}"
        )

    applicable_weights = {season_state: season_weights[season_state] for season_state in season_states}

    total_applicable_weight = sum(applicable_weights.values())

    if total_applicable_weight <= 0:
        raise ValueError(
            "The applicable ignition season weights sum to zero "
            f"for fbp_code={fbp_code}, hex_id={hex_id}. "
            f"Weights={applicable_weights}"
        )

    # Re-normalize over only the states that apply to this FBP code.
    applicable_weights = {season_state: weight / total_applicable_weight for season_state, weight in applicable_weights.items()}

    # Align all curves using ISI as the index.
    aligned_curves = pd.concat(
        {season_state: season_curves[season_state] for season_state in season_states},
        axis=1,
    ).sort_index()

    if aligned_curves.isna().any().any():
        missing_isi = aligned_curves.index[aligned_curves.isna().any(axis=1)].tolist()

        raise ValueError(
            "The SeasonState curves do not contain matching ISI " f"values for fbp_code={fbp_code}. " f"Missing values at ISI={missing_isi}"
        )

    weighted_ros = np.zeros(
        len(aligned_curves),
        dtype=np.float32,
    )

    for season_state, weight in applicable_weights.items():
        weighted_ros += np.float32(weight) * aligned_curves[season_state].to_numpy(dtype=np.float32)

    return weighted_ros


def _build_season_mapping_from_greenup(greenup_path: str | Path) -> dict[str, str]:
    """
    Build a season-name -> ROS SeasonState mapping from a GreenUp CSV.

    The CSV must have columns ``Season`` and ``GreenUp``.  A ``GreenUp``
    value of ``"Yes"`` (case-insensitive) maps to ``"green"``; anything
    else maps to ``"leafless"``.

    Example input
    -------------
    Season,GreenUp
    s1,No
    s2,Yes
    s3, Yes

    Example output
    --------------
    {"s1": "leafless", "s2": "green", "s3": "green"}
    """
    greenup_path = Path(greenup_path)

    if not greenup_path.exists():
        raise FileNotFoundError(f"GreenUp table not found: {greenup_path}")

    df = pd.read_csv(greenup_path)

    required_cols = {"Season", "GreenUp"}
    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(f"Missing required columns in GreenUp table {greenup_path}: " f"{sorted(missing_cols)}")

    df["Season"] = df["Season"].astype(str).str.strip()
    df["GreenUp"] = df["GreenUp"].astype(str).str.strip()

    return {row["Season"]: "green" if row["GreenUp"].lower() == "yes" else "leafless" for _, row in df.iterrows()}


def build_fuel_iros_lookup(
    root_dir: str | Path,
    raw_data_dir: str | Path,
    code_col: str = "fbp_code",
    season_col: str = "SeasonState",
    isi_col: str = "ISI",
    ros_col: str = "ROS",
) -> dict[tuple[int, str], np.ndarray]:
    """
    Construct the complete ROS lookup table once at runtime.

    Lookup format
    -------------
        lookup[(fbp_code, hex_id)] -> ROS vector

    Example
    -------
        ros_lookup[(13, "01")]
        ros_lookup[(13, "05")]

    For each hex, its ignition distribution file is obtained using:

        all_paths = Paths(
            hex_id=hex_id,
            root_dir=root_dir,
        )

        ign_csv = all_paths.ignition_distribution_table(
            hex_id=hex_id
        )

    Parameters
    ----------
    root_dir
        Path to data and fbp_isi_rosi_curves_national_fuel.csv.

    raw_data_dir
        Root directory of hexel data.
    """

    # TODO: save non-seasonal fuel classes only once

    # Read and prepare all ROS curves only once.
    curves_by_code = read_ros_curves(
        ros_csv_path=Path(root_dir) / "fbp_rosi_curves_national_fuel.csv",
        code_col=code_col,
        season_col=season_col,
        isi_col=isi_col,
        ros_col=ros_col,
    )
    ros_lookup: dict[
        tuple[int, str],
        np.ndarray,
    ] = {}

    hex_ids = find_hex_ids(str(raw_data_dir))

    for raw_hex_id in hex_ids:
        hex_id = _normalize_hex_id(raw_hex_id)

        all_paths = Paths(
            hex_id=hex_id,
            root_dir=raw_data_dir,
        )

        greenup_path = all_paths.seasons_greenup_table(hex_id=hex_id)
        hex_season_mapping = _build_season_mapping_from_greenup(greenup_path)

        ignition_distribution_path = Path(all_paths.ignition_distribution_table(hex_id=hex_id))

        season_weights = _read_hex_season_weights(
            distribution_path=ignition_distribution_path,
            season_mapping=hex_season_mapping,
        )

        for fbp_code, season_curves in curves_by_code.items():
            ros_lookup[(fbp_code, hex_id)] = _combine_season_curves(
                fbp_code=fbp_code,
                hex_id=hex_id,
                season_curves=season_curves,
                season_weights=season_weights,
            )

    return ros_lookup


def get_ros_from_lookup(
    ros_lookup: dict[
        tuple[int, str],
        np.ndarray,
    ],
    fbp_code: int,
    hex_id: str | int,
    copy: bool = False,
) -> np.ndarray:
    """
    Retrieve one ROS vector from the precomputed lookup.
    """
    normalized_hex_id = _normalize_hex_id(hex_id)

    key = (
        int(fbp_code),
        normalized_hex_id,
    )

    if key not in ros_lookup:
        raise KeyError("No ROS vector found for " f"fbp_code={key[0]}, hex_id={key[1]}")

    ros_vector = ros_lookup[key]

    if copy:
        return ros_vector.copy()

    return ros_vector


# ros_lookup_table = build_fuel_iros_lookup(root_dir="../burnp3plus/data_samples_v1", raw_data_dir="../burnp3plus")
# print(len(ros_lookup_table))
# ros_vector = get_ros_from_lookup(
#     ros_lookup=ros_lookup_table,
#     fbp_code=1,
#     hex_id="05",
# )

# print(ros_vector)
# print(ros_vector.shape)
