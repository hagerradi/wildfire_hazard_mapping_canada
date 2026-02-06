from abc import ABC, abstractmethod
from collections.abc import Callable
from torch.utils.data import DataLoader, Dataset
import torch
import os
import pandas as pd
import json
import numpy as np
from data_preparation.grid_loader.utils import BURN_COUNT_MAX, BURN_COUNT_MIN, fuel_ranking, get_range_burn_prob
from src.datasets.utils import fill_nan_channel_mean_numpy, one_hot_encode, output_burn_prob_norm
MAX_FUEL_GRID = float(max(fuel_ranking.values()))
MIN_FUEL_GRID = float(min(fuel_ranking.values()))


class DataSource(ABC):
    """Protocol for any feature (spatial grids, weather, wind, etc.)"""
    @abstractmethod
    def get_sample(self, context: dict) -> torch.Tensor:
        """Return the data for a single item"""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Size of the feature vector/number of channels"""
        pass

