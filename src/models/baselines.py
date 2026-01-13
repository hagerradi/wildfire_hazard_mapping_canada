"""
Baselines: Classic U-Net encoder-decoder with optional skip connections.
"""

import torch
import torch.nn as nn

from src.models.utils import get_nbr_model_parameters


class UNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
    ):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output channels
            hidden_features: List of feature maps at each level [64, 128, 256, 512]. Model adjusts accordingly
            use_skip_connections: Whether to use skip connections in the decoder
            use_transpose_conv: use TransposeConv2D in the decoder instead of Upsample with Conv2D
        """
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]

        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # encoder block: downsampling
        in_ch = input_channels
        for h_feature in hidden_features:
            # 4 blocks: n_channelsx64, 64x128, 128x256, 256x512
            self.encoder.append(self._double_conv_block(in_ch, h_feature))
            in_ch = h_feature

        # bottleneck block (bottom of U): 512x1024
        self.bottleneck_layer = self._double_conv_block(hidden_features[-1], hidden_features[-1] * 2)

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
                self.decoder.append(
                    nn.Sequential(
                        upsample_layer,
                        nn.BatchNorm2d(h_feature),
                        nn.LeakyReLU(inplace=True),
                    )
                )
            else:
                self.decoder.append(upsample_layer)

            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.decoder.append(self._double_conv_block(decoder_in_channels, h_feature))

        # output layer
        self.out_conv = nn.Conv2d(hidden_features[0], num_classes, kernel_size=1)

    def _double_conv_block(self, in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        # Encoder: double conv block + maxpool
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x)  # dowsnsample

        # Bottleneck
        x = self.bottleneck_layer(x)

        # reverse skip connections for decoder
        skip_connections = skip_connections[::-1]

        # decoder part
        for i in range(len(self.decoder) // 2):  # (0, 3)
            x = self.decoder[2 * i](x)  # upsample
            if self.use_skip_connections:
                skip_x = skip_connections[i]
                x = torch.cat([skip_x, x], dim=1)
            x = self.decoder[2 * i + 1](x)  # double conv block

        # output layer
        x = self.out_conv(x)
        return x


# TODO: Add test code to verify the model architecture
if __name__ == "__main__":
    model = UNet(input_channels=36, num_classes=1)
    print(model)

    total, trainable = get_nbr_model_parameters(model)

    print(f"Number of total parameters:     {total:,}")
    print(f"Number of trainable parameters: {trainable:,}")

    x = torch.randn(2, 36, 256, 256)  # Batch of 2, 2 channel3, 256x256
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
