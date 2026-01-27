import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionPooling(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (Batch, N, Dim)
        scores = self.attention_net(x) # (Batch, N, 1)
        weights = F.softmax(scores, dim=1)
        weighted_features = x * weights
        return torch.sum(weighted_features, dim=1) # (Batch, Dim)

class MaxPooling(nn.Module):
    def forward(self, x):
        return torch.max(x, dim=1)[0]

class MeanPooling(nn.Module):
    def forward(self, x):
        return torch.mean(x, dim=1)

class WeatherEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int = 8, 
        embed_dim: int = 64,
        pooling_type: str = "attention"
    ):
        super().__init__()

        # 1. Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
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
        
        # 3. Projector
        self.projector = nn.Sequential(
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, embed_dim)
        )

    def forward(self, x):
        B, N, D = x.shape 
        x_flat = x.view(B*N, D)
        x_feats = self.feature_extractor(x_flat) 
        x_feats = x_feats.view(B, N, -1) 
        x_pooled = self.pooler(x_feats)
        return self.projector(x_pooled)