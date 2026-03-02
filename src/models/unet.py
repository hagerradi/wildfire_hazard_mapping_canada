from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.bottlenecks import MultiSourceBottleneck
from src.models.decoders import BaselineDecoder
from src.models.encoders import BaselineEncoder, TabularFeatureEncoder, WindFeatureEncoder
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
    def forward(self, x: torch.Tensor, x_auxillary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
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

    def forward(self, x: torch.Tensor, x_auxillary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
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
        auxillary_input_dims: dict[str, int] | None = None,
        auxillary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] | None = None,
        auxillary_embed_dims: dict[str, int] | None = None,
        auxillary_feature_encoder_poolings: dict[str, str] | None = None,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.num_classes = num_classes
        self.hidden_features = hidden_features if hidden_features else [64, 128, 256, 512]
        self.input_feature_list = input_feature_list
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.auxillary_input_dims: dict[str, int] = auxillary_input_dims or {}
        self.auxillary_hidden_dims: dict[str, list[int] | dict[str, list[int]]] = auxillary_hidden_dims or {}
        self.auxillary_embed_dims: dict[str, int] = auxillary_embed_dims or {}
        self.auxillary_feature_encoder_poolings: dict[str, str] = auxillary_feature_encoder_poolings or {}
        self._build_components()
        self.out_conv = nn.Conv2d(self.hidden_features[0], self.num_classes, kernel_size=1)

    def build_encoder(self) -> nn.Module:
        encoders = nn.ModuleDict()

        features = self.input_feature_list if self.input_feature_list is not None else []

        # Build base spatial grids encoder.
        if "spatial" in features:
            encoders["spatial"] = BaselineEncoder(in_channels=self.input_channels, hidden_features=self.hidden_features)

        # Build encoders for each extra auxillary feature type.
        if self.auxillary_input_dims:
            for name, input_dim in self.auxillary_input_dims.items():
                if name == "wind":
                    hidden_dims = self.auxillary_hidden_dims.get(name, {"mixer": [16], "local": [32, 64, 16], "global": [16]})
                    if isinstance(hidden_dims, dict):
                        encoders[name] = WindFeatureEncoder(
                            in_channels=input_dim, hidden_dims=hidden_dims, embed_dim=self.auxillary_embed_dims.get(name, 16)
                        )
                    continue
                # get the architectural values for each different auxillary encoder
                hidden_dims = self.auxillary_hidden_dims.get(name, [32, 64])
                embed_dim = self.auxillary_embed_dims.get(name, 64)
                pool = self.auxillary_feature_encoder_poolings.get(name, "max")

                if isinstance(hidden_dims, list):
                    encoders[name] = TabularFeatureEncoder(
                        input_dim=input_dim,
                        hidden_dims=hidden_dims,
                        embed_dim=embed_dim,
                        pooling_type=pool,
                    )

        return encoders

    def build_bottleneck(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None.")

        # Get the dims. of all extra auxillary features.
        auxillary_dims: dict[str, int] = {}
        if self.auxillary_input_dims:
            for name in self.auxillary_input_dims.keys():
                auxillary_dims[name] = self.auxillary_embed_dims.get(name, 64)

        return MultiSourceBottleneck(
            in_channels=self.hidden_features[-1], out_channels=self.hidden_features[-1] * 2, aux_dims=auxillary_dims
        )

    def build_decoder(self) -> nn.Module:
        if self.hidden_features is None:
            raise ValueError("Hidden features cannot be None.")

        decoder = BaselineDecoder(
            hidden_features=self.hidden_features,
            use_skip_connections=self.use_skip_connections,
            use_transpose_conv=self.use_transpose_conv,
            use_activation_after_upsampling=self.use_activation_after_upsampling,
        )
        return decoder

    def forward(self, x: torch.Tensor, x_auxillary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        """
        Args:
            x: Spatial input (B, C, H, W) (torch.Tensor)
            x_auxillary: Dict. of auxiliary inputs {'weather': (B, N, D), ...} (dict[str, torch.Tensor] | None)
        """

        skip_connections = []
        tabular_embeddings = []

        # Spatial encoder path.
        if "spatial" in self.encoder:  # type: ignore
            x, skip_connections = self.encoder["spatial"](x)  # type: ignore

        # Extra auxillary encoders path.
        x_wind = None
        if self.auxillary_input_dims and x_auxillary is not None:
            for name in self.auxillary_input_dims.keys():
                if name in x_auxillary:
                    encoder_aux = self.encoder[name]  # type: ignore
                    encoder_emb = encoder_aux(x_auxillary[name])
                    if name == "wind":
                        x_wind = encoder_emb
                        continue
                    tabular_embeddings.append(encoder_emb)

        # Bottleneck path: concat. all tabular embeds.
        x_fused_tabular = None
        if len(tabular_embeddings) > 0:
            x_fused_tabular = torch.cat(tabular_embeddings, dim=1)
        x = self.bottleneck(x, x_fused_tabular, x_wind)

        # Decoder and head.
        x = self.decoder(x, skip_connections)
        return self.out_conv(x)
