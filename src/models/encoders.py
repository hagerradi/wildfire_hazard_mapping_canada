from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.utils import double_conv_block


class EncoderBase(nn.Module, ABC):
    """Abstract base class for encoders.
    forward(x) -> (out, skips) where skips is List[Tensor] (shallow -> deep).
    Must set self.out_channels to the number channels of `out`.
    """

    def __init__(self):
        super().__init__()
        self.in_channels: int
        self.hidden_features: list | None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError


class BaselineEncoder(EncoderBase):
    """
    Encoder implementing the downsampling path of your UNet.
    forward(x) -> (bottleneck, skips) with skips shallow -> deep.
    Sets self.out_channels to bottleneck channels.
    """

    def __init__(self, in_channels: int, hidden_features: list):
        super().__init__()
        self.hidden_features = hidden_features

        self.layers = nn.ModuleList()
        in_ch = in_channels
        for h_feature in self.hidden_features:
            self.layers.append(double_conv_block(in_ch, h_feature))
            in_ch = h_feature

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        return x, skip_connections
