# Data preparation pipeline

Step 1:
[Only for modelling approach 2] To generate the rasters per season and cause:

```bash
python -m data_preparation.generate_season_cause_output --root_dir="../yan_bp3"
```
This saves the rasters in the original data folders under `outputs`

Step 2: Process hexel data into multiple square windows, which will be our data samples:
```bash
python -m data_preparation.process_hexels_into_grids --root_dir="../yan_bp3" --modelling_approach=2 --output_type="count" --win_h=128 --win_w=128 --overlap_ratio=0.2 --mask_threshold=0.5
```

Step 3: Create training, validation and test splits.
```bash
python -m data_preparation.split_data --data_dir="../yan_bp3/data_samples_approach_2" --val_hex_id 16 --test_hex_id 41
```
