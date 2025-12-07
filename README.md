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

### Data preparation

For all the data preparation steps, refer to [the following section](data_preparation/README.md).

### Training & Inference

TODO
