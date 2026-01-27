import torch
import torch.nn as nn

from src.models.components.encoders import WeatherEncoder


class DoubleConvBlock(nn.Module):
    """
    (Conv3x3 -> BN -> ReLU) x 2
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class StandardBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels)

    def forward(self, x, x_weather=None):
        return self.conv(x)


class WeatherFusionBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, weather_input_dim: int, weather_embed_dim: int, pooling_type: str):
        super().__init__()
        # 1. Standard Spatial Processing
        self.conv = DoubleConvBlock(in_channels, out_channels)

        # 2. Weather Encoding
        self.weather_encoder = WeatherEncoder(input_dim=weather_input_dim, embed_dim=weather_embed_dim, pooling_type=pooling_type)

        # 3. Fusion Projector
        # Projects (Spatial_Channels + Weather_Embed) -> Spatial_Channels
        self.fusion_projector = nn.Sequential(
            nn.Conv2d(out_channels + weather_embed_dim, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x, x_weather):
        if x_weather is None:
            raise ValueError("WeatherFusionBottleneck requires x_weather input.")

        # Spatial path
        x = self.conv(x)

        # Weather path
        weather_emb = self.weather_encoder(x_weather)  # (B, embed_dim)

        # Tile / Broadcast
        B, C, H, W = x.shape
        weather_tiled = weather_emb.view(B, -1, 1, 1).expand(-1, -1, H, W)

        # Concatenate & Project
        x_fused = torch.cat([x, weather_tiled], dim=1)
        return self.fusion_projector(x_fused)
