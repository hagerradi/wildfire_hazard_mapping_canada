"""
Multi Encoder Models: Contains a Modular U-Net with Weather Injection.
"""

from typing import List, Optional

import torch
import torch.nn as nn

from src.models.components.bottlenecks import DoubleConvBlock, StandardBottleneck, WeatherFusionBottleneck

class FusionUNet(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        hidden_features: Optional[List[int]] = None,
        use_skip_connections: bool = True,
        use_transpose_conv: bool = False,
        use_activation_after_upsampling: bool = False,
        weather_input_dim: Optional[int] = None,
        weather_embed_dim: int = 64,
        pooling_type: str = "attention",
        use_spatial_encoder: bool = True,
        bottleneck_spatial_size: int = 8,  # Required if encoder is disabled (128px / 2^4 = 8px)
    ):
        super().__init__()

        if hidden_features is None:
            hidden_features = [64, 128, 256, 512]

        # 1. Logic to handle "Weather Only" mode
        self.use_spatial_encoder = use_spatial_encoder
        self.bottleneck_spatial_size = bottleneck_spatial_size

        # If no encoder, we must disable skip connections
        if not self.use_spatial_encoder:
            use_skip_connections = False

        self.use_skip_connections = use_skip_connections
        self.use_transpose_conv = use_transpose_conv
        self.use_activation_after_upsampling = use_activation_after_upsampling
        self.weather_input_dim = weather_input_dim

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # 2. Build Encoder (if enabled)
        if self.use_spatial_encoder:
            in_ch = input_channels
            for h_feature in hidden_features:
                self.encoder.append(DoubleConvBlock(in_ch, h_feature))
                in_ch = h_feature
        
        # 3. Bottleneck
        # If encoder exists, input is hidden_features[-1] (e.g. 512)
        # If no encoder, we will generate a dummy input of this same size
        bottleneck_in = hidden_features[-1]
        bottleneck_out = hidden_features[-1] * 2

        # Explicit type hint prevents MyPy error
        self.bottleneck: nn.Module

        if self.weather_input_dim:
            if weather_input_dim is None:
                raise ValueError("weather_input_dim cannot be None when using weather bottleneck")

            self.bottleneck = WeatherFusionBottleneck(
                in_channels=bottleneck_in,
                out_channels=bottleneck_out,
                weather_input_dim=weather_input_dim,
                weather_embed_dim=weather_embed_dim,
                pooling_type=pooling_type,
            )
        else:
            self.bottleneck = StandardBottleneck(bottleneck_in, bottleneck_out)

        # 4. Decoder
        for h_feature in reversed(hidden_features):
            # Upsampling
            if self.use_transpose_conv:
                upsample = nn.Sequential(nn.ConvTranspose2d(h_feature * 2, h_feature, kernel_size=2, stride=2))
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
            # If skips are OFF (because encoder is OFF), in_channels is just h_feature
            # If skips are ON, in_channels is h_feature * 2
            decoder_in_channels = h_feature * 2 if self.use_skip_connections else h_feature
            self.decoder.append(DoubleConvBlock(decoder_in_channels, h_feature))

        # Output
        self.out_conv = nn.Conv2d(hidden_features[0], num_classes, kernel_size=1)

    def forward(self, x: Optional[torch.Tensor], x_weather: Optional[torch.Tensor] = None) -> torch.Tensor:
        skip_connections = []

        # --- A. Encoder Path ---
        if self.use_spatial_encoder:
            if x is None:
                raise ValueError("Model initialized with spatial encoder, but input 'x' is None.")
                
            for encoder_block in self.encoder:
                x = encoder_block(x)
                skip_connections.append(x)
                x = self.maxpool(x)
        else:
            # --- B. No Encoder Path (Generation Mode) ---
            # We need to construct a "dummy" feature map to start the bottleneck
            # We use x_weather to get the batch size and device
            if x_weather is None:
                raise ValueError("Weather Only mode requires x_weather input.")
            
            B = x_weather.shape[0]
            # Dimensions: (Batch, 512, 8, 8) - Matches output of the last encoder block
            enc_channels = self.bottleneck.conv.net[0].in_channels # Retrieve input channels dynamically or use hidden_features[-1]
            H, W = self.bottleneck_spatial_size, self.bottleneck_spatial_size
            
            # Create zeros on the correct GPU device
            x = torch.zeros((B, enc_channels, H, W), device=x_weather.device)

        # --- Bottleneck ---
        # If 'x' is zeros, the conv layers act as bias generators, 
        # and then weather is fused onto it.
        x = self.bottleneck(x, x_weather)

        # --- Decoder ---
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