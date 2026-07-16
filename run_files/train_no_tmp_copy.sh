#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=unet_fi_ros
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=10:59:00
#SBATCH --mem-per-cpu=40Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

set -euo pipefail

# Capture the first argument, default to the common pipeline config.
CONFIG_FILE=${1:-configs/bp_common_input_pipeline.yaml}
TRAIN_ARGS=${TRAIN_ARGS:-}
EVAL_ARGS=${EVAL_ARGS:-}
RUN_HEXEL_EVAL=${RUN_HEXEL_EVAL:-1}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate
LOGGER_ENABLED=$(python - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open() as handle:
    config = yaml.safe_load(handle)

print(str(config.get("logger", {}).get("enabled", True)).lower())
PY
)

if [[ "$LOGGER_ENABLED" == "true" && -z "${COMET_API_KEY:-}" ]]; then
    echo "ERROR: COMET_API_KEY must be exported when logger.enabled=true." >&2
    exit 1
fi

echo "Using data.root_dir directly from config: $CONFIG_FILE"

echo "Running training with config: $CONFIG_FILE"
read -r -a TRAIN_ARG_ARRAY <<< "$TRAIN_ARGS"
python -m src.train \
    --config="$CONFIG_FILE" \
    "${TRAIN_ARG_ARRAY[@]}"
