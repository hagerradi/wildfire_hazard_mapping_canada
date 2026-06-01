#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=bp_channel_gen
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=05:59:00
#SBATCH --mem-per-cpu=20Gb
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a100:1

set -euo pipefail

CONFIG_FILE=${1:-configs/default_v1_full_data_bp_full_config_kl_ccc_hexpairrank_fullsupport_bpzero.yaml}
OUTPUT_DIR=${2:-experiments/bp_full_config_v3_kl_ccc_hexpairrank_fullsupport_bpzero_restricted_postprocess_all_splits}
MASK_SCOPE=${MASK_SCOPE:-actual}
SPLITS=${SPLITS:-"train_indices.csv val_indices.csv test_indices.csv"}
CHECKPOINT=${CHECKPOINT:-}

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

CMD=(
  python -m src.generate_bp_prediction_channel
  --config "$CONFIG_FILE"
  --output_dir "$OUTPUT_DIR"
  --mask_scope "$MASK_SCOPE"
  --splits $SPLITS
  --no_save_patch_predictions
)

if [[ -n "$CHECKPOINT" ]]; then
  CMD+=(--checkpoint "$CHECKPOINT")
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
