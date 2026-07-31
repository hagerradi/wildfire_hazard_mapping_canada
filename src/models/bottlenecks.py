import torch
import torch.nn as nn

from src.models.encoders import append_coord_channels
from src.models.utils import double_conv_block


class MultiSourceBottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, aux_dims: dict[str, int], use_coordconv: bool = False):
        super().__init__()
        self.use_coordconv = use_coordconv
        # Primary spatial processing
        spatial_in_channels = in_channels + 2 if self.use_coordconv else in_channels
        self.spatial_conv = double_conv_block(spatial_in_channels, out_channels)

        # Calculate concatenation depth: spatial_out + sum of all auxiliary vector dims
        total_depth = out_channels + sum(aux_dims.values())

        # Projection layer to bring fused features back to U-Net bottleneck width
        self.fusion_projector = nn.Sequential(
            nn.Conv2d(total_depth, out_channels, kernel_size=1), nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor, x_tabular: torch.Tensor | None = None, x_wind: torch.Tensor | None = None) -> torch.Tensor:
        if self.use_coordconv:
            x = append_coord_channels(x)
        x = self.spatial_conv(x)
        if x_tabular is None and x_wind is None:
            return x
        B, C, H, W = x.shape
        concat_list = [x]
        if x_tabular is not None:
            # Tile: (B, D) -> (B, D, 1, 1) -> (B, D, H, W)
            x_tab_tiled = x_tabular.view(B, -1, 1, 1).expand(-1, -1, H, W)
            concat_list.append(x_tab_tiled)
        # Concatenate along the channel dimension (dim=1)
        if x_wind is not None:
            concat_list.append(x_wind)

        x_fused = torch.cat(concat_list, dim=1)

        return self.fusion_projector(x_fused)
