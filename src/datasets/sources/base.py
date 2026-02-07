from abc import ABC, abstractmethod

import torch


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
