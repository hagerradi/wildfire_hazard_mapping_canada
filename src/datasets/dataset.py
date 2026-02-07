import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.config import DataConfig, DataSourceConfig
from src.datasets.registry import Registry
from src.datasets.sources.base import DataSource
from src.datasets.sources.grids import GridSource
from src.datasets.sources.weather import WeatherSource
from src.datasets.transforms import get_transforms
from src.utils import seed_everything, seed_worker

SOURCE_MAPPING = {"grid": GridSource, "weather": WeatherSource}


class MultiSourceDataset(Dataset):
    """
    Composable dataset that orchestrates data loading from multiple sources.

    Assignes index management to a `Registry` and feature extraction to
    individual `DataSource` objects. To optimize I/O, raw data files are
    memory-mapped once per sample in `__getitem__` and injected into a shared
    context, allowing multiple sources (example: grid, weather) to read
    from the same file efficiently.
    """

    def __init__(self, registry: Registry, sources: dict[str, DataSource]):
        """
        Args:
        registry (Registry): Manages file paths and metadata.
        sources (dict[str, DataSource]): A dictionary mapping output keys
            (example: 'grid', 'weather') to their respective data sources.
        """
        self.registry = registry
        self.sources = sources

    def __getitem__(self, idx):
        context = self.registry.get_context(idx)
        context["data"] = np.load(context["file_path"], mmap_mode="r")  # Load once, distribute where needed
        sample = {}
        for name, source in self.sources.items():
            sample[name] = source.get_sample(context)
        return sample

    def __len__(self):
        return len(self.registry)


def build_dataset(config: DataConfig, csv_name: str, modelling_approach: str = "1") -> MultiSourceDataset:
    # 1. Build registry
    root_dir = config.root_dir
    filename_col = config.filename_col
    valid_mask_threshold = config.valid_mask_threshold
    registry = Registry(csv_name=csv_name, root_dir=root_dir, filename_col=filename_col, valid_mask_threshold=valid_mask_threshold)

    # 2. Build sources
    sources = {}
    for source_conf in config.sources:
        # Inject global parameters
        params = source_conf.params.copy()
        params["root_dir"] = root_dir
        params["modelling_approach"] = modelling_approach
        # Setup transforms
        transform = get_transforms(source_conf)
        params.pop("transform", None)
        params.pop("augmentation_prob", None)
        # Get class from mapping
        source_class = SOURCE_MAPPING[source_conf.name]
        # Instantiate
        sources[source_conf.name] = source_class(**params, transform=transform)
    return MultiSourceDataset(registry=registry, sources=sources)


def get_train_val_dataloader(config: DataConfig, modelling_approach: str = "1", seed: int = 42) -> tuple[DataLoader, DataLoader]:
    """
    Creates and returns a DataLoader with deterministic shuffling
    """
    batch_size = config.batch_size
    num_workers = config.num_workers
    train_split = config.train_split
    val_split = config.val_split

    seed_everything(seed=seed, deterministic=True)
    g = torch.Generator()
    train_dataset = build_dataset(config, csv_name=config.train_split)
    val_dataset = build_dataset(config, csv_name=config.val_split)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, worker_init_fn=seed_worker, generator=g
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g
    )

    return train_dataloader, val_dataloader


# Very basic implementation of dataset from scratch (not using config)
if __name__ == "__main__":
    registry = Registry(csv_name="train_indices.csv", root_dir="yan_bp3/data_samples_approach_1_weather")
    grid_source = GridSource(
        root_dir="yan_bp3/data_samples_approach_1_weather", feature_names_list=["ignition_grid", "fuel_grid", "elevation_grid"]
    )
    weather_source = WeatherSource(
        csv_name="weather_table.csv",
        root_dir="yan_bp3/data_samples_approach_1_weather",
        feature_names_list=["temp", "rh", "prec", "ffmc", "dmc", "dc", "isi", "bui"],
    )
    dataset = MultiSourceDataset(registry=registry, sources={"grid": grid_source, "weather": weather_source})
    dataloader = DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))
    print(batch)
