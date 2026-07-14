#!/bin/bash
#SBATCH --job-name=cf_fuel_iROS
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=unkillable
#SBATCH --ntasks=1
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=6Gb
#SBATCH --gres=gpu:a100:1

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

python -m src.evaluate_counterfactual --config configs/counterfactual_fuel.yaml --overwrite
python -m src.datasets.postprocessing.counterfactual_fuel_intervention_map \
    --config configs/counterfactual_fuel.yaml \
    --experiment_dir experiments/counterfactual_fuel_hex16 \
    --scenario remove_barriers_adjacent_modal --endpoint bp --hex_id 16
python -m src.datasets.postprocessing.counterfactual_change_distribution \
    --experiment_dir experiments/counterfactual_fuel_hex16 --endpoint bp
python -m src.datasets.postprocessing.counterfactual_change_distribution \
    --experiment_dir experiments/counterfactual_fuel_hex16 --endpoint fi
