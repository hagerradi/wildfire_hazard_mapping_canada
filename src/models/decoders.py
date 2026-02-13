from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.utils import double_conv_block


class DecoderBase(nn.Module, ABC):
    """Abstract base class for decoders.
    forward(x, skips) -> out
    skips should be ordered shallow -> deep (same order as encoder produced).
    Must accept the bottleneck feature x and apply upsampling + skip merges.
    """

    def __init__(self):
        super().__init__()
        self.out_channels: int | None

    @abstractmethod
    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError


class BaselineDecoder(DecoderBase):
    """
    Baseline Decoder for Unet
    """

    def __init__(
        self,
        hidden_features: list[int] | None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
    ):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.hidden_features = list(hidden_features)
        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling

        self.layers = nn.ModuleList()

        # decoder block: upsampling
        for h_feature in reversed(hidden_features):
            # 4 downsampling blocks: 1024x512, 512x256, 256x128, 128x64
            if self.use_transpose_conv:
                # standard method: ConvTranspose2d
                upsample_layer = nn.Sequential(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2))
            else:
                # upsample + Conv2d
                upsample_layer = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(h_feature * 2, h_feature, kernel_size=3, padding=1, bias=False),
                )

            if self.use_activation_after_upsampling:
                self.layers.append(
                    nn.Sequential(
                        upsample_layer,
                        nn.BatchNorm2d(h_feature),
                        nn.LeakyReLU(inplace=True),
                    )
                )
            else:
                self.layers.append(upsample_layer)

            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.layers.append(double_conv_block(decoder_in_channels, h_feature))

    def forward(self, x: torch.Tensor, skip_connections: list[torch.Tensor]) -> torch.Tensor:
        """
        x: bottleneck feature
        skip_connections: shallow -> deep (as produced by EncoderUNet)
        returns: decoded feature map (channels == hidden_features[0])
        """
        # reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        for i in range(len(self.layers) // 2):  # (0, 3)
            x = self.layers[2 * i](x)  # upsample
            if self.use_skip_connections:
                skip_x = skip_connections[i]
                x = torch.cat([skip_x, x], dim=1)
            x = self.layers[2 * i + 1](x)  # double conv block

        return x
