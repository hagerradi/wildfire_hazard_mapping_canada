# Inference Module

End-to-end inference pipeline for wildfire risk prediction on hexels.


## Quick Start

### Command Line

```bash
# Option 1: Run inference using values from config.yaml
python -m inference.run_ai_surrogate_model_hexel_inference

# Option 2: Override hexel ID and data preparation
python -m inference.run_ai_surrogate_model_hexel_inference --hex_id="05" --prepare_data=False

# Option 3: Run inference on all hexels
python -m inference.run_ai_surrogate_model_hexel_inference --hex_id="all" --prepare_data=True

# Option 4: Run inference using all CLI commands
python -m inference.run_ai_surrogate_model_hexel_inference --data_dir=/path/to/hexel/data --checkpoint_path=/path/to/model/best.pth --save_dir=/path/to/output/ --hex_id="all" --batch_size=64 --num_workers=8 --prepare_data=True
```
**CLI Arguments**

Note: All arguments follow values from `config.yaml` but if CLI arguments are provided, they will override the config values.

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | `inference/config.yaml` |
| `--data_dir` | Directory containing hexel data | From config |
| `--checkpoint_path` | Path to model checkpoint | From config |
| `--save_dir` | Directory to save predictions and visualizations | From config |
| `--hex_id` | (str) Hexel ID (str) to process | From config |
| `--prepare_data` | Run data preparation step | `True` |
| `--batch_size` | Batch size for inference | From config |
| `--num_workers` | DataLoader workers | From config |



## Output

Note: We assume `save_dir` is set to `outputs/` for the following paths.

- **Normalized Patch Predictions**: Saved as `.npy` file at `outputs/predictions_patches/predictions_hexel_{hex_id}.npy` with shape `(N, C, H, W)`
- **Reconstructed Hexel**: Post-processed reconstructed hexel grid of predicted burn probabilities and geospatial profile from ground truth saved as GeoTIFF at `outputs/`
- **Visualizations**: Comparison plots of ground truth vs predictions for each hexel saved at `outputs/predicted_hexel/`
- **Logs**: Written to both console and `inference/run_hexel_inference.log`

## Configuration

Edit `config.yaml`:

```yaml
# Paths
data_dir: "/path/to/hexel/data"
checkpoint_path: "/path/to/model/best.pth"
save_dir: "/path/to/output/"  # Predictions saved as predictions_hexel_{hex_id}.npy

# Hexel configuration
hex_id: "02"

# Data preparation settings
prepare_data: True      # Set true to run data prep

# Inference settings
batch_size: 32
num_workers: 4
```

## Components

The module is split into two components:

### 1. BurnRiskPredictor (`predictor.py`)
Pure inference class that wraps the PyTorch model. Handles only tensor operations - no data loading or hexel-specific logic.

### 2. run_ai_surrogate_model_hexel_inference (`run_ai_surrogate_model_hexel_inference.py`)
Orchestrates the full pipeline:
1. Data preparation (patch splitting, tabular data processing)
2. Dataset creation from checkpoint config
3. Prediction loop using `BurnRiskPredictor`
4. Post-processing and visualization
5. Saving results with log file output.


### Programmatic Usage

```python
from inference import BurnRiskPredictor, run_pipeline

# Option 1: Full end-to-end pipeline
predicted_hexel_grid, grid_profile = run_pipeline(
    checkpoint_path="experiments/best_model/best.pth",
    data_dir="data/",
    hex_id="02",
    prepare_data=True,
    save_dir="outputs/",  # Saves as predictions_hexel_02.npy
)

# Option 2: Just the predictor (for custom pipelines or serving)
predictor = BurnRiskPredictor.from_checkpoint(
    checkpoint_path="experiments/best_model/best.pth",
    spatial_channels=10,
    auxiliary_input_dims={"weather": 7, "fire_size": 5},
)
predicted_hexel_grid, grid_profile = predictor(spatial_batch, auxiliary_batch)
```

### Folder Structure

```
inference/
├── data_dir/                                 # Hexel data directory (raw hexel data, fire size distribution)
├── predictor.py                              # Pure ML engine (tensor-in, tensor-out)
├── run_ai_surrogate_model_hexel_inference.py # Orchestration (data prep, dataset, prediction loop, post-processing)
├── config.yaml                               # Configuration file
└── run_hexel_inference.log                   # Output logs
```

## Data Requirements

The `data_dir` should contain:
- `hex{hex_id}/` - Raw hexel data directory
- `df_fire_fru.csv` - Fire size distribution

If `prepare_data=True`, the pipeline will:

1. Build weather tables
2. Process fire size distributions
3. Split hexel into patches
4. Generate metadata CSV
