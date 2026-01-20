#!/bin/bash
#SBATCH --job-name=unet_full_data
#SBATCH --output=job_output.txt
#SBATCH --error=job_error.txt
#SBATCH --ntasks=1
#SBATCH --time=13:59:00
#SBATCH --mem-per-cpu=50Gb
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1

source .venv/bin/activate
export COMET_API_KEY=$COMET_API_KEY
python -m src.train
