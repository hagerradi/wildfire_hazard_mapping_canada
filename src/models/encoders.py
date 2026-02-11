from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from src.models.utils import double_conv_block
import torch.nn.functional as F

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

    def __init__(self, in_channels: int, hidden_features: list[int] | None):
        super().__init__()
        if hidden_features is None:
            raise ValueError("Hidden features cannot be None")
        self.hidden_features = hidden_features

        self.layers = nn.ModuleList()
        in_ch = in_channels
        for h_feature in self.hidden_features:
            self.layers.append(double_conv_block(in_ch, h_feature))
            in_ch = h_feature

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            skip_connections.append(x)
            x = self.maxpool(x)
        return x, skip_connections


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
    def __init__(self, input_dim: int, embed_dim: int = 64, pooling_type: str = "max"):
        super().__init__()
        self.embed_dim = embed_dim
        
        # 1. Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
        )

        # 2. Select Pooler
        self.pooling_type = pooling_type.lower()
        if self.pooling_type == "attention":
            self.pooler = AttentionPooling(input_dim=64)
        elif self.pooling_type == "max":
            self.pooler = MaxPooling()
        elif self.pooling_type == "mean":
            self.pooler = MeanPooling()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # 3. Projector to final embedding dimension
        self.projector = nn.Sequential(
            nn.Linear(64, 64), 
            nn.LeakyReLU(inplace=True), 
            nn.Linear(64, self.embed_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (Batch, N_observations, Dimensions)
        B, N, D = x.shape
        
        # Flatten Batch and N to process all observations through the same weights
        x_flat = x.view(B * N, D)
        # Extract features
        x_feats = self.feature_extractor(x_flat) # (B*N, 64)
        # Reshape back to (B, N, 64) for pooling
        x_feats = x_feats.view(B, N, -1)         
        # Collapse N dimension (e.g., summarize all weather stations into one vector)
        x_pooled = self.pooler(x_feats) # (B, 64)
        # Final projection to match the bottleneck's expected auxiliary dimension
        return self.projector(x_pooled) # (B, embed_dim)