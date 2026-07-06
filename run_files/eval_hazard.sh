#!/bin/bash
#SBATCH --job-name=hazard_eval
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=unkillable
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --mem-per-cpu=8Gb
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:a100:1

set -euo pipefail

# Usage: sbatch run_files/eval_hazard.sh configs/<your_hazard_config>.yaml
# Extra CLI args (e.g. --metrics_only --skip_plots) can be passed via EVAL_ARGS:
#   EVAL_ARGS="--metrics_only" sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
# Override mask_scope/save_dir/root_dir to run actual vs buffer variants from one config without output collisions:
#   EVAL_ARGS="--mask_scope buffer_only --root_dir /path/to/buffer_root --save_dir experiments/hazard_eval/buffer_only" \
#     sbatch run_files/eval_hazard.sh configs/hazard_eval_common_input_pipeline.yaml
CONFIG_FILE=${1:-configs/hazard_eval_common_input_pipeline.yaml}
EVAL_ARGS=${EVAL_ARGS:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

RUN_CONFIG_FILE="$CONFIG_FILE"

if [[ -n "${SLURM_TMPDIR:-}" ]]; then
    echo "Using SLURM_TMPDIR for staged dataset: ${SLURM_TMPDIR}"

    ORIGINAL_DATA_ROOT_DIR=$(python - "$CONFIG_FILE" "$EVAL_ARGS" <<'PY'
import shlex
import sys
from pathlib import Path
import yaml
config_path = Path(sys.argv[1])
eval_args = shlex.split(sys.argv[2])
with config_path.open() as handle:
    config = yaml.safe_load(handle)
root_dir = config["root_dir"]
for idx, arg in enumerate(eval_args):
    if arg == "--root_dir" and idx + 1 < len(eval_args):
        root_dir = eval_args[idx + 1]
    elif arg.startswith("--root_dir="):
        root_dir = arg.split("=", 1)[1]
print(Path(root_dir).resolve())
PY
)

    STAGE_PARENT="${SLURM_TMPDIR}/nrcan_wildfireriskmapping_data"
    STAGED_DATA_ROOT_DIR="${STAGE_PARENT}/$(basename "$ORIGINAL_DATA_ROOT_DIR")_${SLURM_JOB_ID:-$$}"
    mkdir -p "$STAGE_PARENT"
    trap 'rm -rf "$STAGED_DATA_ROOT_DIR"' EXIT

    echo "Staging eval subset from root_dir:"
    echo "  from: ${ORIGINAL_DATA_ROOT_DIR}"
    echo "  to:   ${STAGED_DATA_ROOT_DIR}"
    df -h "$SLURM_TMPDIR" || true

    STAGE_FILE_LIST="${SLURM_TMPDIR}/$(basename "$ORIGINAL_DATA_ROOT_DIR")_${SLURM_JOB_ID:-$$}_files.txt"
    python - "$CONFIG_FILE" "$ORIGINAL_DATA_ROOT_DIR" "$STAGE_FILE_LIST" <<'PY'
import csv
import sys
from pathlib import Path

import yaml

config_path = Path(sys.argv[1])
root_dir = Path(sys.argv[2])
file_list_path = Path(sys.argv[3])

with config_path.open() as handle:
    config = yaml.safe_load(handle)

def safe_relative_path(value, source):
    raw_value = str(value or "").strip()
    path = Path(raw_value)
    if not raw_value or path.is_absolute() or ".." in path.parts or "\n" in raw_value or "\r" in raw_value:
        raise ValueError(f"Unsafe staged path from {source}: {raw_value!r}")
    return path.as_posix()

test_split = config.get("test_split", "test_indices.csv")
relative_paths = {safe_relative_path(path.name, "root_dir") for path in root_dir.iterdir() if path.is_file()}
with (root_dir / test_split).open(newline="") as handle:
    reader = csv.DictReader(handle)
    if reader.fieldnames is None or "filename" not in reader.fieldnames:
        raise ValueError(f"{root_dir / test_split} must contain a 'filename' column.")
    relative_paths.update(safe_relative_path(row["filename"], f"{root_dir / test_split}:filename") for row in reader)

missing_paths = sorted(path for path in relative_paths if not (root_dir / path).is_file())
if missing_paths:
    preview = ", ".join(missing_paths[:5])
    raise FileNotFoundError(f"Missing {len(missing_paths)} staged input files under {root_dir}: {preview}")

with file_list_path.open("w") as handle:
    for path in sorted(relative_paths):
        handle.write(f"{path}\n")

print(f"Prepared staging list with {len(relative_paths)} files: {file_list_path}", flush=True)
PY

    if command -v rsync >/dev/null 2>&1; then
        rsync -a --files-from="$STAGE_FILE_LIST" "${ORIGINAL_DATA_ROOT_DIR}/" "${STAGED_DATA_ROOT_DIR}/"
    else
        mkdir -p "$STAGED_DATA_ROOT_DIR"
        while IFS= read -r relative_path; do
            mkdir -p "${STAGED_DATA_ROOT_DIR}/$(dirname "$relative_path")"
            cp -a "${ORIGINAL_DATA_ROOT_DIR}/${relative_path}" "${STAGED_DATA_ROOT_DIR}/${relative_path}"
        done <"$STAGE_FILE_LIST"
    fi

    echo "Staged dataset size:"
    du -sh "$STAGED_DATA_ROOT_DIR" || true

    RUN_CONFIG_FILE="${SLURM_TMPDIR}/$(basename "${CONFIG_FILE%.yaml}")_slurm_tmpdir.yaml"
    python - "$CONFIG_FILE" "$RUN_CONFIG_FILE" "$STAGED_DATA_ROOT_DIR" <<'PY'
import sys
from pathlib import Path
import yaml
source_config = Path(sys.argv[1])
run_config = Path(sys.argv[2])
staged_root = Path(sys.argv[3])
with source_config.open() as handle:
    config = yaml.safe_load(handle)
config["root_dir"] = str(staged_root)
with run_config.open("w") as handle:
    yaml.safe_dump(config, handle, sort_keys=False)
print(f"Wrote staged config: {run_config}")
print(f"Using persistent raw_data_dir for evaluation: {config['raw_data_dir']}")
PY
else
    echo "SLURM_TMPDIR is not set; using config root_dir directly."
fi

echo "Running hazard evaluation with config: $RUN_CONFIG_FILE"
EVAL_ARG_ARRAY=()
while IFS= read -r -d '' arg; do
    EVAL_ARG_ARRAY+=("$arg")
done < <(python - "$EVAL_ARGS" <<'PY'
import shlex
import sys

for arg in shlex.split(sys.argv[1]):
    sys.stdout.write(arg + "\0")
PY
)
if [[ -n "${STAGED_DATA_ROOT_DIR:-}" ]]; then
    EVAL_ARG_ARRAY+=(--root_dir "$STAGED_DATA_ROOT_DIR")
fi
python -m src.evaluate_hazard --config="$RUN_CONFIG_FILE" "${EVAL_ARG_ARRAY[@]}"
