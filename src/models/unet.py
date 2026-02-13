from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.bottlenecks import MultiSourceBottleneck
from src.models.decoders import BaselineDecoder
from src.models.encoders import BaselineEncoder, TabularFeatureEncoder
from src.models.utils import double_conv_block


class UNetBase(nn.Module, ABC):
    """Abstract base class for UNet models."""

    def __init__(self):
        super().__init__()

        self.input_channels: int
        self.num_classes: int
        self.hidden_features: list[int] | None
        self.input_feature_list: list | None
        self.use_skip_connections: bool
        self.use_transpose_conv: bool
        self.use_activation_after_upsampling: bool

        self.encoder: nn.Module
        self.bottleneck: nn.Module
        self.decoder: nn.Module

    @abstractmethod
    def build_encoder(self) -> nn.Module | nn.ModuleDict:
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
    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class BaselineUNet(UNetBase):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list[int] | None = None,
        input_feature_list: list | None = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
    ):
        super().__init__()
        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]
        if input_feature_list is None:
            input_feature_list = ["spatial"]

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.input_feature_list = input_feature_list
        self._build_components()
        # output layer
        self.out_conv = nn.Conv2d(self.hidden_features[0], self.num_classes, kernel_size=1)

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

    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None) -> torch.Tensor:
        x, skip_connections = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x, skip_connections)
        x = self.out_conv(x)
        return x


class MultiSourceUNet(UNetBase):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list[int] | None = None,
        input_feature_list: list | None = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
        tabular_input_dim: int | None = None,
        tabular_embed_dim: int | None = None,
        tabular_feature_encoder_pooling: str | None = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features if hidden_features else [64, 128, 256, 512]
        self.input_feature_list = input_feature_list
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.tabular_input_dim = tabular_input_dim
        self.tabular_embed_dim = tabular_embed_dim
        self.tabular_feature_encoder_pooling = tabular_feature_encoder_pooling
        self._build_components()
        self.out_conv = nn.Conv2d(self.hidden_features[0], self.num_classes, kernel_size=1)

    def build_encoder(self) -> nn.Module:
        encoders = nn.ModuleDict()
        features = self.input_feature_list if self.input_feature_list is not None else []

        if "spatial" in features:
            encoders["spatial"] = BaselineEncoder(in_channels=self.input_channels, hidden_features=self.hidden_features)
        if "tabular" in features:
            if self.tabular_input_dim is None:
                raise ValueError("tabular_input_dim cannot be None when 'tabular' is in input_feature_list")
            pooling = self.tabular_feature_encoder_pooling if self.tabular_feature_encoder_pooling is not None else "max"
            encoders["tabular"] = TabularFeatureEncoder(input_dim=self.tabular_input_dim, pooling_type=pooling)
        return encoders

    def build_bottleneck(self) -> nn.Module:
        features = self.input_feature_list if self.input_feature_list is not None else []

        aux_dims: dict[str, int] = {}

        if "tabular" in features:
            if self.tabular_embed_dim is None:
                raise ValueError("tabular_embed_dim cannot be None when using tabular features")
            aux_dims["tabular"] = self.tabular_embed_dim

        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None")

        return MultiSourceBottleneck(in_channels=self.hidden_features[-1], out_channels=self.hidden_features[-1] * 2, aux_dims=aux_dims)

    def build_decoder(self) -> nn.Module:
        decoder = BaselineDecoder(
            hidden_features=self.hidden_features,
            use_skip_connections=self.use_skip_connections,
            use_transpose_conv=self.use_transpose_conv,
            use_activation_after_upsampling=self.use_activation_after_upsampling,
        )
        return decoder

    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None) -> torch.Tensor:
        skip_connections = []
        tabular_emb = None

        # 1. Spatial Path
        if "spatial" in self.encoder:  # type: ignore
            x, skip_connections = self.encoder["spatial"](x)  # type: ignore

        # 2. Tabular Path
        if "tabular" in self.encoder and x_tabular is not None:  # type: ignore
            tabular_emb = self.encoder["tabular"](x_tabular)  # type: ignore

        # 3. Bottleneck
        # Passes the spatial map and the (encoded or None) tabular features
        x = self.bottleneck(x, tabular_emb)

        # 4. Decoder and Head
        x = self.decoder(x, skip_connections)
        return self.out_conv(x)
