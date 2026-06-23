from abc import ABC, abstractmethod
from functools import lru_cache

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils import conv_block, double_conv_block


@lru_cache(maxsize=64)
def _coord_grids(height: int, width: int, device: str, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    y_coord = torch.linspace(-1.0, 1.0, height, device=torch.device(device), dtype=dtype).view(1, 1, height, 1)
    y_coord = y_coord.expand(1, 1, height, width)
    x_coord = torch.linspace(-1.0, 1.0, width, device=torch.device(device), dtype=dtype).view(1, 1, 1, width)
    x_coord = x_coord.expand(1, 1, height, width)
    return y_coord, x_coord


def append_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """Append patch-local y/x coordinates normalized to [-1, 1]."""
    batch_size, _, height, width = x.shape
    y_coord, x_coord = _coord_grids(height, width, str(x.device), x.dtype)
    y_coord = y_coord.expand(batch_size, 1, height, width)
    x_coord = x_coord.expand(batch_size, 1, height, width)
    return torch.cat([x, y_coord, x_coord], dim=1)


class EncoderBase(nn.Module, ABC):
    """Abstract base class for encoders.
    forward(x) -> (out, skips) where skips is List[Tensor] (shallow -> deep).
    Must set self.out_channels to the number channels of `out`.
    """

    def __init__(self):
        super().__init__()
        self.in_channels: int
        self.hidden_features: list | None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        raise NotImplementedError


class BaselineEncoder(EncoderBase):
    """
    Encoder implementing the downsampling path of your UNet.
    forward(x) -> (bottleneck, skips) with skips shallow -> deep.
    Sets self.out_channels to bottleneck channels.
    """

    def __init__(self, in_channels: int, hidden_features: list[int] | None, use_coordconv: bool = False):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.hidden_features = hidden_features
        self.use_coordconv = use_coordconv

        self.layers = nn.ModuleList()
        in_ch = in_channels
        for h_feature in self.hidden_features:
            if self.use_coordconv:
                in_ch += 2
            self.layers.append(double_conv_block(in_ch, h_feature))
            in_ch = h_feature

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections = []
        for layer in self.layers:
            if self.use_coordconv:
                x = append_coord_channels(x)
            x = layer(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        return x, skip_connections


class iROSEncoder(nn.Module):
    """
    Independently encode the iROS curve at every pixel.

    Input:
        x: (B, num_isi_values, H, W)

    Output:
        embedding: (B, embed_dim, H, W)
    """

    def __init__(
        self, iros_mean: torch.Tensor, iros_std: torch.Tensor, in_channels: int = 18, hidden_dim: int = 16, embed_dim: int = 4
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.embed_dim = embed_dim
        if iros_mean.numel() != in_channels:
            raise ValueError(f"Expected {in_channels} mean values, got {iros_mean.numel()}")

        if iros_std.numel() != in_channels:
            raise ValueError(f"Expected {in_channels} std values, got {iros_std.numel()}")

        self.in_channels = in_channels
        self.embed_dim = embed_dim

        self.iros_mean: torch.Tensor
        self.iros_std: torch.Tensor
        self.register_buffer(
            "iros_mean",
            iros_mean.reshape(1, in_channels, 1, 1),
        )

        self.register_buffer(
            "iros_std",
            iros_std.reshape(1, in_channels, 1, 1).clamp_min(1e-6),
        )

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=hidden_dim),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                hidden_dim,
                embed_dim,
                kernel_size=1,
                bias=True,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: iROS curves with shape (B, 18, H, W).

        Returns:
            Per-pixel embeddings with shape (B, embed_dim, H, W).
        """
        if x.ndim != 4:
            raise ValueError(f"Expected a 4D tensor (B, C, H, W), got {tuple(x.shape)}")

        if x.shape[1] != self.in_channels:
            raise ValueError(f"Expected {self.in_channels} iROS channels, " f"got {x.shape[1]}")
        x = torch.log1p(x.clamp_min(0))
        x = (x - self.iros_mean) / self.iros_std

        return self.encoder(x)


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.attention_net = nn.Sequential(nn.Linear(input_dim, 32), nn.Tanh(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (Batch, N, Dim)
        scores = self.attention_net(x)  # (Batch, N, 1)
        weights = F.softmax(scores, dim=1)
        weighted_features = x * weights
        return torch.sum(weighted_features, dim=1)  # (Batch, Dim)


class MaxPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.max(x, dim=1)[0]


class MeanPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.mean(x, dim=1)


class TabularFeatureEncoder(nn.Module):
    """
    Encoder for vector/tabular data (like weather).
    Processes inputs of shape (B, N, D) and returns an embedding (B, embed_dim).
    """

    def __init__(self, input_dim: int, hidden_dims: list[int] | None = None, embed_dim: int = 64, pooling_type: str = "max"):
        super().__init__()
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 64]
        self.embed_dim = embed_dim

        # 1. Dynamic feature extractor (to handle any number of layers)
        layers: list[nn.Module] = []
        in_ch = input_dim
        for h_dim in self.hidden_dims:
            layers.append(nn.Linear(in_ch, h_dim))
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.ReLU(inplace=True))
            in_ch = h_dim

        self.feature_extractor = nn.Sequential(*layers)

        # Get last layer size to pass to pooler
        last_hidden_dim = self.hidden_dims[-1] if len(self.hidden_dims) > 0 else input_dim

        # 2. Select Pooler
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == "attention":
            self.pooler = AttentionPooling(input_dim=last_hidden_dim)  # type: ignore
        elif self.pooling_type == "max":
            self.pooler = MaxPooling()  # type: ignore
        elif self.pooling_type == "mean":
            self.pooler = MeanPooling()  # type: ignore
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # 3. Projector to final embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(last_hidden_dim, last_hidden_dim), nn.LeakyReLU(inplace=True), nn.Linear(last_hidden_dim, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, N_observations, Dimensions)
        B, N, D = x.shape

        # Flatten Batch and N to process all observations through the same weights
        x_flat = x.view(B * N, D)
        # Extract features
        x_feats = self.feature_extractor(x_flat)  # (B*N, last_dim)
        # Reshape back to (B, N, last_dim) for pooling
        x_feats = x_feats.view(B, N, -1)
        # Collapse N dimension (e.g., summarize all weather samples into one vector)
        x_pooled = self.pooler(x_feats)  # (B, last_dim)
        # Final projection to match the bottleneck's expected auxiliary dimension
        return self.projector(x_pooled)  # (B, embed_dim)


class WindFeatureEncoderMixer(nn.Module):
    def __init__(self, in_channels: int = 16, hidden_dims: dict[str, list[int]] | None = None, embed_dim: int = 16):
        super().__init__()

        self.hidden_dims = hidden_dims if hidden_dims is not None else {"mixer": [16], "local": [32, 64, 16], "global": [16]}
        self.embed_dim = embed_dim

        # 1. Mixer: 16 -> 16 (1x1 selection)
        mixer_layers: list[nn.Module] = []
        in_ch = in_channels
        for h_dim in self.hidden_dims["mixer"]:
            mixer_layers.append(nn.Conv2d(in_ch, h_dim, kernel_size=1))
            in_ch = h_dim
        self.mixer = nn.Sequential(*mixer_layers)

        # 2. Local Path: Hourglass (128 -> 8 spatial | 16 -> 64 -> 16 channels)
        local_path_layers: list[nn.Module] = []
        group_norm_kernel = [4, 8, 4]
        kernel_sizes = [4, 3, 3]
        strides = [4, 2, 2]
        paddings = [0, 1, 1]
        in_ch = self.hidden_dims["mixer"][-1]
        for i, h_dim in enumerate(self.hidden_dims["local"]):
            local_path_layers.append(nn.Conv2d(in_ch, h_dim, kernel_size=kernel_sizes[i], stride=strides[i], padding=paddings[i]))
            local_path_layers.append(nn.GroupNorm(group_norm_kernel[i], h_dim))
            local_path_layers.append(nn.ReLU(inplace=True))
            in_ch = h_dim
        self.local_path = nn.Sequential(*local_path_layers)

        # 3. Global Path: (Spearman stability)
        # 128 -> 8 | 16 -> 16 channels
        self.global_path = nn.Sequential(
            nn.AvgPool2d(kernel_size=16, stride=16),  # 128 -> 8 (Fixed, no 'adaptive' logic)
            nn.Conv2d(self.hidden_dims["mixer"][-1], self.hidden_dims["global"][0], kernel_size=1),
        )

        # 4. Final Fusion (16 local + 16 global = 32 -> 16)
        self.fusion = nn.Sequential(
            nn.Conv2d(self.hidden_dims["local"][-1] + self.hidden_dims["global"][-1], self.embed_dim, kernel_size=1),
            nn.GroupNorm(4, self.embed_dim),
            nn.ReLU(inplace=True),
        )

        self.scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mixer(x)
        local_feat = self.local_path(x)  # (B, 16, 8, 8)
        global_feat = self.global_path(x)  # (B, 16, 8, 8)
        combined = torch.cat([local_feat, global_feat], dim=1)  # (B, 32, 8, 8)
        return self.fusion(combined) * self.scale


class WindFeatureEncoderSpatial(nn.Module):
    """
    Encodes 128x128 wind grids with `in_channels` channels down to 8x8x`embed_dim`.
    Assuming input shape is (Batch, Channels, Height, Width) -> (B, in_channels, 128, 128).
    """

    def __init__(self, in_channels=16, hidden_dims=None, embed_dim=64):
        super().__init__()

        if hidden_dims is None:
            raise ValueError("Hidden Dim for Wind Encoder cannot be None")
        self.hidden_dims = hidden_dims

        self.layers = nn.ModuleList()
        in_ch = in_channels
        for h_feature in self.hidden_dims:  # 16,32, 64
            self.layers.append(conv_block(in_ch, h_feature))
            in_ch = h_feature
        self.feature_extractor = nn.Sequential(*self.layers)
        # Step 4: 16x16 -> 8x8
        self.projector = conv_block(self.hidden_dims[-1], embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(x)
        x = self.projector(x)
        return x  # Output is (B, embed_dim, 8, 8)
