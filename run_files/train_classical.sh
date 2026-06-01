#!/bin/bash
#SBATCH --job-name=classical_lgbm
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=08:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=16

set -euo pipefail

CONFIG_FILE=${1:-configs/default_v1_full_data_bp_full_config.yaml}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

RUN_CONFIG_FILE="$CONFIG_FILE"
ORIGINAL_DATA_ROOT_DIR=""

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    echo "Using SLURM_TMPDIR for staged dataset: ${SLURM_TMPDIR}"

    ORIGINAL_DATA_ROOT_DIR=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
with config_path.open() as handle:
    config = yaml.safe_load(handle)

print(Path(config["data"]["root_dir"]).resolve())
PY
)

    STAGE_PARENT="${SLURM_TMPDIR}/nrcan_wildfireriskmapping_data"
    STAGED_DATA_ROOT_DIR="${STAGE_PARENT}/$(basename "$ORIGINAL_DATA_ROOT_DIR")"
    mkdir -p "$STAGE_PARENT"

    echo "Staging data.root_dir:"
    echo "  from: ${ORIGINAL_DATA_ROOT_DIR}"
    echo "  to:   ${STAGED_DATA_ROOT_DIR}"
    df -h "$SLURM_TMPDIR" || true

    if command -v rsync >/dev/null 2>&1; then
        rsync -a "${ORIGINAL_DATA_ROOT_DIR}/" "${STAGED_DATA_ROOT_DIR}/"
    else
        mkdir -p "$STAGED_DATA_ROOT_DIR"
        cp -a "${ORIGINAL_DATA_ROOT_DIR}/." "$STAGED_DATA_ROOT_DIR/"
    fi

    echo "Staged dataset size:"
    du -sh "$STAGED_DATA_ROOT_DIR" || true

    RUN_CONFIG_FILE="${SLURM_TMPDIR}/$(basename "${CONFIG_FILE%.yaml}")_classical_tmpdir.yaml"
    python - "$CONFIG_FILE" "$RUN_CONFIG_FILE" "$STAGED_DATA_ROOT_DIR" <<'PY'
import sys
from pathlib import Path

import yaml

source_config = Path(sys.argv[1])
run_config = Path(sys.argv[2])
staged_root = Path(sys.argv[3])

with source_config.open() as handle:
    config = yaml.safe_load(handle)

config["data"]["root_dir"] = str(staged_root)

with run_config.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)

print(f"Wrote staged config: {run_config}")
print(f"Using persistent raw_data_dir for normalization/evaluation: {config['data']['raw_data_dir']}")
PY
else
    echo "SLURM_TMPDIR is not set; using config data.root_dir directly."
fi

CLASSICAL_ARGS=${CLASSICAL_ARGS:-}
echo "Running classical LightGBM baseline with config: $RUN_CONFIG_FILE"
echo "Extra args: ${CLASSICAL_ARGS}"
python -m src.train_classical \
    --config="$RUN_CONFIG_FILE" \
    --num-threads="${SLURM_CPUS_PER_TASK:-16}" \
    ${CLASSICAL_ARGS}
