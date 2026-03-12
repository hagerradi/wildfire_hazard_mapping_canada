## Canada-Wide WildFire Risk Mapping

### Installation & Setup

Install `uv`: https://docs.astral.sh/uv/getting-started/installation.

Clone the repository:
```bash
   git clone https://github.com/milatechtransfer/nrcan_wildfireriskmapping.git
   cd nrcan_wildfireriskmapping
```
Then, create/update the environment from the lockfile:
```bash
   uv sync
```
To activate the environment:
```bash
source .venv/bin/activate
```
To add a new dependency/package into the codebase:
```bash
uv add <PACKAGE>
```
This will automatically update the `pyproject.toml` to include the new package, as well as regenerate the updated `uv.lock` and install the new package into the `.venv`.

To activate the pre-commit hooks, run:
```bash
uv run pre-commit install
```

### Logging with Comet

To set up the logging with Comet, add your API key via:
```bash
export COMET_API_KEY=<YOUR_KEY>
```

### Data preparation

For all the data preparation steps, refer to [the following section](data_preparation/README.md).

### Training

```python -m src.train --config=configs/default_v1.yaml```

#### On the cluster:
To launch a job on the cluster, use the script `run_files/train.sh`.
Steps:
1. `export COMET_API_KEY=YOUR_KEY`
2. Run `uv sync`, if needed
3. Run with the desired config filename `sbatch run_files/train.sh configs/default_v1_full_data.yaml`. By default, it uses `configs/default_v1.yaml`.

### Inference

To visualize predictions and/or save visualizations, add the optional flags `--visualize_predictions` and/or `--save_visualizations`, respectively.

```python -m src.evaluate_hexels --config=configs/default_v1.yaml```

If we want to use center crop for stitching predictions, make sure step 2 in data_preparation is run with `overlap_with_halo=64` and `overlap_ratio=None`

Then, you should use `--stitch_mode=center_crop` to run `src.evaluate_hexels`


### Generate full Canada map of hexels

To generate the full Canada hexel map of targets and/or predictions, run the following script (see --help for more args. information):

```python -m src.datasets.postprocessing.full_map.generate_full_hexel_map --data-dir data/ --scale "log" --show_hex_borders --output "experiments/full_canada_map.png"```
