from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union  # noqa: UP035

import torch
import torch.nn as nn


class UNetBase(nn.Module, ABC):
    """Abstract base class for unets."""

    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list[int] | None = None,
        feature_list: list | None = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        if feature_list is None:
            feature_list = ["spatial"]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.feature_list = feature_list

        self.encoder: nn.Module
        self.bottlenecks: nn.Module
        self.decoder: nn.Module

    @abstractmethod
    def forward(self, x_spatial: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
