from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.decoders import BaselineDecoder
from src.models.encoders import BaselineEncoder
from src.models.utils import double_conv_block


class UNetBase(nn.Module, ABC):
    """Abstract base class for unets."""

    def __init__(self):
        super().__init__()

        self.input_channels: int
        self.num_classes: int
        self.hidden_features: list[int] | None
        self.feature_list: list | None
        self.use_skip_connections: bool
        self.use_transpose_conv: bool
        self.use_activation_after_upsampling: bool

        self.encoder: nn.Module
        self.bottleneck: nn.Module
        self.decoder: nn.Module

        # self._build_components()

    @abstractmethod
    def build_encoder(self) -> nn.Module:
        """Return an EncoderBase-derived module (or any module whose forward returns (bottleneck, skips))."""
        raise NotImplementedError

    @abstractmethod
    def build_bottleneck(self) -> nn.Module:
        """Return the bottleneck module (nn.Module) applied to the deepest feature map."""
        raise NotImplementedError

    @abstractmethod
    def build_decoder(self) -> nn.Module:
        """Return a DecoderBase-derived module (or any module that accepts (bottleneck, skips) -> features)."""
        raise NotImplementedError

    def _build_components(self) -> None:
        """
        Top-level hook that calls the build methods.
        Subclasses may override build_* methods or override _build_components itself.
        """
        self.encoder = self.build_encoder()
        self.bottleneck = self.build_bottleneck()
        self.decoder = self.build_decoder()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class BaselineUNet(UNetBase):
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
        # output layer
        self.out_conv = nn.Conv2d(self.hidden_features[0], self.num_classes, kernel_size=1)

        self._build_components()

    def build_encoder(self) -> nn.Module:
        encoder = BaselineEncoder(in_channels=self.input_channels, hidden_features=self.hidden_features)
        return encoder

    def build_bottleneck(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        return double_conv_block(self.hidden_features[-1], self.hidden_features[-1] * 2)

    def build_decoder(self) -> nn.Module:
        decoder = BaselineDecoder(
            hidden_features=self.hidden_features,
            use_skip_connections=self.use_skip_connections,
            use_transpose_conv=self.use_transpose_conv,
            use_activation_after_upsampling=self.use_activation_after_upsampling,
        )
        return decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.out_conv(x)
        return x
