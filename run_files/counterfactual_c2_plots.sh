#!/bin/bash
#SBATCH --job-name=cf_c2_plots
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long-cpu
#SBATCH --ntasks=1
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=16Gb

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs
source .venv/bin/activate

python -m src.datasets.postprocessing.counterfactual_fuel_intervention_map \
    --config configs/counterfactual_fuel.yaml \
    --experiment_dir experiments/counterfactual_fuel_hex16 \
    --scenario remove_barriers_fixed_c2 --endpoint bp --hex_id 16
python -m src.datasets.postprocessing.counterfactual_change_distribution \
    --experiment_dir experiments/counterfactual_fuel_hex16 --scenario remove_barriers_fixed_c2 --endpoint bp
python -m src.datasets.postprocessing.counterfactual_change_distribution \
    --experiment_dir experiments/counterfactual_fuel_hex16 --scenario remove_barriers_fixed_c2 --endpoint fi
