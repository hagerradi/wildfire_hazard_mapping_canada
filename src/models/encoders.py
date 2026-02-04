from abc import ABC, abstractmethod
from collections.abc import Sequence

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
        self.out_channels: int | None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError


class BaselineEncoder(EncoderBase):
    """
    Encoder implementing the downsampling path of your UNet.
    forward(x) -> (bottleneck, skips) with skips shallow -> deep.
    Sets self.out_channels to bottleneck channels.
    """

    def __init__(self, in_channels: int = 1, hidden_features: Sequence[int] | None = None):
        super().__init__()
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        self.hidden_features = list(hidden_features)

        self.layers = nn.ModuleList()
        in_ch = in_channels
        for h_feature in self.hidden_features:
            self.layers.append(double_conv_block(in_ch, h_feature))
            in_ch = h_feature

        # bottleneck double conv: last_hidden -> last_hidden * 2
        self.bottleneck = double_conv_block(self.hidden_features[-1], self.hidden_features[-1] * 2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # expose output channels (bottleneck channels)
        self.out_channels = self.hidden_features[-1] * 2

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections = []
        for stage in self.layers:
            x = stage(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        x = self.bottleneck(x)
        return x, skip_connections
