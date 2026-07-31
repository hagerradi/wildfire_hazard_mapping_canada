# src/datasets/sources/__init__.py

from src.datasets.sources.base import DataSource
from src.datasets.sources.grids import GridSource
from src.datasets.sources.spatialized_tabular import SpatializedTabularSource
from src.datasets.sources.tabular import TabularSource

# This defines what gets imported when someone does: from src.datasets.sources import *
__all__ = ["GridSource", "TabularSource", "SpatializedTabularSource", "DataSource"]
