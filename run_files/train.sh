#!/bin/bash
##SBATCH --mail-type=all
##SBATCH --mail-user=name@mila.quebec
#SBATCH --job-name=unet_full_data
#SBATCH --output=logs/job_%x_%j.out
#SBATCH --error=logs/job_%x_%j.err
#SBATCH --partition=long
#SBATCH --ntasks=1
#SBATCH --time=05:59:00
#SBATCH --mem-per-cpu=10Gb
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1

# Capture the first argument, default to 'configs/default_v1.yaml' if empty
CONFIG_FILE=${1:-configs/default_v1.yaml}

mkdir -p logs
source .venv/bin/activate
export COMET_API_KEY=$COMET_API_KEY
echo "Running training with config: $CONFIG_FILE"
python -m src.train --config="$CONFIG_FILE"
echo "Running evaluation with config: $CONFIG_FILE"
python -m src.evaluate_hexels --config=configs/default_v1.yaml
