"""
Multi Encoder Models: Contains a Modular U-Net with Weather Injection.
"""

import torch
import torch.nn as nn
from src.models.components.bottlenecks import DoubleConvBlock, StandardBottleneck, WeatherFusionBottleneck

class WeatherUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: list = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
        weather_input_dim: int = None,
        weather_embed_dim: int = 64,
        pooling_type: str = "attention"
    ):
        """
        Args:
            input_channels: Number of input channels
            num_classes: Number of output channels
            hidden_features: List of feature maps at each level [64, 128, 256, 512]. Model adjusts accordingly
            use_skip_connections: Whether to use skip connections in the decoder
            use_transpose_conv: use TransposeConv2D in the decoder instead of Upsample with Conv2D
            weather_input_dim: If None, acts like a standard U-Net.
            weather_embed_dim: Dimension of weather vector after encoding.
            pooling_type: 'attention', 'mean', or 'max' for the DeepSet encoder.
        """
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]

        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.weather_input_dim = weather_input_dim

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 1. Encoder
        in_ch = input_channels
        for h_feature in hidden_features:
            self.encoder.append(DoubleConvBlock(in_ch, h_feature))
            in_ch = h_feature

        # 2. Bottleneck
        bottleneck_in = hidden_features[-1]
        bottleneck_out = hidden_features[-1] * 2
        
        if self.weather_input_dim:
            self.bottleneck = WeatherFusionBottleneck(
                in_channels=bottleneck_in,
                out_channels=bottleneck_out,
                weather_input_dim=weather_input_dim,
                weather_embed_dim=weather_embed_dim,
                pooling_type=pooling_type
            )
        else:
            self.bottleneck = StandardBottleneck(bottleneck_in, bottleneck_out)

        # 3. Decoder
        for h_feature in reversed(hidden_features):
            # Upsampling
            if self.use_transpose_conv:
                upsample = nn.Sequential(
                    nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2)
                )
            else:
                upsample = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                    nn.Conv2d(h_feature * 2, h_feature, kernel_size=3, padding=1, bias=False),
                )
            
            if self.use_activation_after_upsampling:
                upsample.add_module("bn_up", nn.BatchNorm2d(h_feature))
                upsample.add_module("relu_up", nn.LeakyReLU(inplace=True))

            self.decoder.append(upsample)

            # Conv Block
            decoder_in_channels = h_feature * 2 if use_skip_connections else h_feature
            self.decoder.append(DoubleConvBlock(decoder_in_channels, h_feature))

        # Output
        self.out_conv = nn.Conv2d(hidden_features[0], num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor, x_weather: torch.Tensor = None) -> torch.Tensor:
        skip_connections = []

        # Encoder
        for encoder_block in self.encoder:
            x = encoder_block(x)
            skip_connections.append(x)
            x = self.maxpool(x)

        # Bottleneck (Polymorphic: handles weather injection if configured)
        x = self.bottleneck(x, x_weather)

        # Decoder
        skip_connections = skip_connections[::-1]
        for i in range(len(self.decoder) // 2):
            x = self.decoder[2 * i](x)
            
            if self.use_skip_connections:
                skip_x = skip_connections[i]
                if x.shape != skip_x.shape:
                    x = nn.functional.interpolate(x, size=skip_x.shape[2:])
                x = torch.cat([skip_x, x], dim=1)
            
            x = self.decoder[2 * i + 1](x)

        return self.out_conv(x)