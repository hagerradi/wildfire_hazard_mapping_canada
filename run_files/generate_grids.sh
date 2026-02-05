#!/bin/bash
#SBATCH --job-name=grid_gen
#SBATCH --output=logs/array_%x_%A_%a.log
#SBATCH --array=0-52 # hard-coded since we know there's 53 hexel subdirs
#SBATCH --ntasks=1
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2

mkdir -p logs
source .venv/bin/activate

# Robust calculation of Num Tasks (Max Index - Min Index + 1)
TASK_ID=${SLURM_ARRAY_TASK_ID:-0}
# Default to 1 if not running in array
if [ -n "$SLURM_ARRAY_TASK_MAX" ]; then
    NUM_TASKS=$((SLURM_ARRAY_TASK_MAX - SLURM_ARRAY_TASK_MIN + 1))
else
    NUM_TASKS=1
fi

echo "Starting Worker $TASK_ID / $NUM_TASKS"

python -m data_preparation.process_hexels_into_grids \
    --root_dir="/network/projects/amlrt/nrcan_wildfires/full_data/yan_bp3" \
    --save_dir="INSERT PATH HERE" \
    --modelling_approach=1 \
    --output_type="prob" \
    --win_h=128 \
    --win_w=128 \
    --overlap_ratio=0.2 \
    --weather_sampling="weather_zone_id" \
    --is_array_job \
    --task_id=$TASK_ID \
    --num_tasks=$NUM_TASKS
