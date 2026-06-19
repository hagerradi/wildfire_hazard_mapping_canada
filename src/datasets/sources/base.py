from abc import ABC, abstractmethod
from typing import Any


class DataSource(ABC):
    """Protocol for any feature (spatial grids, weather, wind, etc.)"""

    @abstractmethod
    def get_sample(self, context: dict) -> Any:
        """Return the data for a single item"""
        pass

    @property
    @abstractmethod
    def input_dim(self) -> int:
        """Size of the feature vector/number of channels"""
        pass
