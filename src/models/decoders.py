from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
    def forward(self, x: torch.Tensor, skips: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError
