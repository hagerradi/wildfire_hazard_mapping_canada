# Data preparation pipeline

Step 1:
[Only for modelling approach 2] To generate the rasters per season and cause:

```bash
python -m data_preparation.generate_season_cause_output --root_dir="../yan_bp3"
```
This saves the rasters in the original data folders under `outputs`

Step 2: Process hexel data into multiple square windows, which will be our data samples:

Note: If you want to run this in the cluster using SLURM array jobs, you can modify the `run_files/generate_grid_data.sh` by changing the save directory path and run the following in the terminal (from the main directory)

```bash
sbatch run_files/generate_grid_data.sh
```

Instead, you can run the following on an interactive node:
```bash
python -m data_preparation.process_hexels_into_grids --root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA"  --save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v1" --modelling_approach=1 --win_h=128 --win_w=128 --overlap_ratio=0.2
```

Step 3: Create training, validation and test splits.

- Run the `get_stratified_data_split` function in `data_preparation/utils.py` to run stratified sampling over the available hex_ids. This will give a train, val, test split with 37,5,5 hexels in each respectively
- Now, verify this split looking at the geographical map and find if the split is well distributed
- Once the split is finalized, use the decided split to obtain the train/val/test csvs
- Finally, run the following with the decided splits

```bash
python -m data_preparation.split_data --data_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_v1" --val_hex_id 02 23 33 18 46 --test_hex_id 01 12 39 16 49
```

Step 4: Create tabular files (weather + fire-size)

To produce the sequential weather table and the fire-size distribution table used by the model. Both tables are mapped to patches via the fire weather zone ID, so your grids must include that ID.

To build the tabular files, run the following:


```bash
python -m data_preparation.process_tabular_data \
	--root_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA" \
	--save_dir="/network/projects/amlrt/nrcan_wildfires/data/full_data_bp3plus/canada_bp3+_2026_MILA/data_samples_approach_1" \
	--weather_output_file="weather_table_processed.csv" \
	--fire_size_input_file="df_fire_fru.csv" \
	--fire_size_output_file="df_fire_fru_processed.csv"
```

Notes: If you used modelling approach 2, set `--save_dir` to `data_samples_approach_2`. The `process_tabular_data` script will look for the fire-size file in `--root_dir` first, then in `--save_dir`; ensure `df_fire_fru.csv` is present in one of those places.
