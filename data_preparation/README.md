# Data preparation pipeline

## Generate Hexel Grids

Step 1:
[Only for modelling approach 2] To generate the rasters per season and cause:

```bash
python -m data_preparation.generate_season_cause_output --root_dir="../yan_bp3"
```
This saves the rasters in the original data folders under `outputs`

Step 2: Process hexel data into multiple square windows, which will be our data samples:
```bash
python -m data_preparation.process_hexels_into_grids --root_dir="../yan_bp3" --modelling_approach=2 --output_type="count" --win_h=128 --win_w=128 --overlap_ratio=0.2
```
Note: If you want to run this in the cluster using SLURM array jobs (much quicker), you can modify the `run_files/generate_grid.sh` by changing the save directory path and run the following in the terminal (from the main directory)

```
sbatch run_files/generate_grids.sh
```

Step 3: Create training, validation and test splits.

- Run the `get_stratified_data_split` function in `data_preparation/utils.py` to run stratified sampling over the available hex_ids. This will give a train, val, test split with 37,5,5 hexels in each respectively
- Now, verify this split looking at the geographical map and find if the split is well distributed
- Once the split is finalized, use the decided split to obtain the train/val/test csvs
- Finally, run the following with the decided splits

```bash
python -m data_preparation.split_data --data_dir="../yan_bp3/data_samples_approach_2" --val_hex_id 02 23 33 18 46 --test_hex_id 01 12 39 16 49
```


Step 4: To create the weather aggregated weather table used to sample from in the weather fused unet, run the following. 

```
python build_aggregated_weather_table.py --root_dir="/network/projects/amlrt/nrcan_wildfires/full_data/yan_bp3" --file_name="weather_table.csv" --save_dir="<INSERT PATH HERE>"
```