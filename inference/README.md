# Inference Module

End-to-end inference pipeline for wildfire risk prediction on individual hexels.

## Components

The module is split into two components:

### 1. BurnRiskPredictor (`predictor.py`)
Pure inference class that wraps the PyTorch model. Handles only tensor operations - no data loading or hexel-specific logic.

### 2. run_hexel_inference (`run_hexel_inference.py`)
Orchestrates the full pipeline:
1. Data preparation (patch splitting, tabular data processing)
2. Dataset creation from checkpoint config
3. Prediction loop using `BurnRiskPredictor`
4. Saving results

### Folder Structure

```
inference/
├── predictor.py           # Pure ML engine (tensor-in, tensor-out)
├── run_hexel_inference.py # Orchestration (data prep, dataset, prediction loop)
├── config.yaml            # Configuration file
└── inference.log          # Output logs
```


## Quick Start

### Command Line

```bash
# Run inference with no overrides
python -m inference.run_hexel_inference --config=inference/config.yaml

# Override hexel ID and data preparation
python -m inference.run_hexel_inference --config=inference/config.yaml --hex_id=05 --prepare_data=False
```

### Programmatic Usage

```python
from inference import BurnRiskPredictor, run_pipeline

# Option 1: Full end-to-end pipeline
predictions = run_pipeline(
    checkpoint_path="experiments/best_model/best.pth",
    root_dir="data/",
    hex_id="02",
    prepare_data=True,
    save_path="outputs/predictions.npy",
)

# Option 2: Just the predictor (for custom pipelines or serving)
predictor = BurnRiskPredictor.from_checkpoint(
    checkpoint_path="experiments/best_model/best.pth",
    spatial_channels=10,
    auxiliary_input_dims={"weather": 7, "fire_size": 5},
)
predictions = predictor(spatial_batch, auxiliary_batch)
```

## Configuration

Edit `config.yaml`:

```yaml
# Paths
data_dir: "/path/to/hexel/data"
checkpoint_path: "/path/to/model/best.pth"
output_path: "/path/to/predictions.npy"

# Hexel configuration
hex_id: "02"

# Inference settings
batch_size: 32
num_workers: 4

# Data preparation settings
prepare_data: false      # Set true to run data prep
modelling_approach: 1    # 1 = joint season-cause, 2 = separate
win_h: 128               # Patch height
win_w: 128               # Patch width
overlap_ratio: 0.2       # Overlap between patches
```

Note: `modelling_approach`, `win_h`, `win_w`, and `overlap_ratio` has to be consistent with training configuration.

## CLI Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--config` | Path to YAML config file | `inference/config.yaml` |
| `--hex_id` | Hexel ID to process (overrides config) | From config |
| `--prepare_data` | Run data preparation step | `True` |
| `--batch_size` | Batch size for inference | From config |
| `--num_workers` | DataLoader workers | From config |

## Output

- **Predictions**: Saved as `.npy` file at `output_path` with shape `(N, C, H, W)`
- **Logs**: Written to both console and `inference/inference.log`

## Data Requirements

The `data_dir` should contain:
- `hex{hex_id}/` - Raw hexel data directory
- `df_fire_fru.csv` - Fire size distribution

If `prepare_data=True`, the pipeline will:

1. Build weather tables
2. Process fire size distributions
3. Split hexel into patches
4. Generate metadata CSV
