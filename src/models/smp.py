from typing import Any

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.encoders import append_coord_channels


class SMPDensePredictionModel(nn.Module):
    """Adapter around segmentation_models_pytorch dense prediction models.

    Auxiliary tabular features are pooled per patch and concatenated as constant
    spatial channels. Auxiliary grid features are resized and concatenated as
    additional image channels.
    """

    def __init__(
        self,
        *,
        architecture: str,
        encoder_name: str,
        encoder_weights: str | None,
        input_channels: int,
        num_classes: int,
        use_coordconv: bool = False,
        auxiliary_input_dims: dict[str, int] | None = None,
        auxiliary_feature_encoder_poolings: dict[str, str] | None = None,
        smp_kwargs: dict[str, Any] | None = None,
    ):
        super().__init__()
        if input_channels <= 0:
            raise ValueError(f"input_channels must be positive, got {input_channels}.")

        kwargs = dict(smp_kwargs or {})
        if "aux_params" in kwargs:
            raise ValueError("SMP classification auxiliary heads are not supported by this dense-prediction trainer.")

        self.use_coordconv = use_coordconv
        self.auxiliary_input_dims = auxiliary_input_dims or {}
        self.auxiliary_feature_encoder_poolings = auxiliary_feature_encoder_poolings or {}
        auxiliary_channels = sum(self.auxiliary_input_dims.values())
        model_input_channels = input_channels + auxiliary_channels + (2 if use_coordconv else 0)
        self.model = smp.create_model(
            arch=architecture,
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=model_input_channels,
            classes=num_classes,
            **kwargs,
        )

    def _pool_tabular_auxiliary(self, name: str, aux: torch.Tensor) -> torch.Tensor:
        pooling = self.auxiliary_feature_encoder_poolings.get(name, "mean").lower()
        if pooling == "mean":
            return aux.mean(dim=1)
        if pooling == "max":
            return aux.max(dim=1).values
        raise ValueError(f"SMP auxiliary channel fusion supports 'mean' or 'max' pooling, got '{pooling}' for '{name}'.")

    def _auxiliary_channels(self, x_auxiliary: dict[str, torch.Tensor] | None, spatial_shape: tuple[int, int]) -> list[torch.Tensor]:
        if not self.auxiliary_input_dims:
            return []
        if x_auxiliary is None:
            raise ValueError(f"Missing auxiliary inputs required by SMP model: {sorted(self.auxiliary_input_dims)}")

        aux_channels = []
        for name, expected_channels in self.auxiliary_input_dims.items():
            if name not in x_auxiliary:
                raise ValueError(f"Missing auxiliary input '{name}' required by SMP model.")
            aux = x_auxiliary[name]
            if aux.ndim == 3:
                pooled = self._pool_tabular_auxiliary(name, aux)
                if pooled.shape[1] != expected_channels:
                    raise ValueError(f"Auxiliary input '{name}' has {pooled.shape[1]} channels, expected {expected_channels}.")
                aux = pooled[:, :, None, None].expand(-1, -1, *spatial_shape)
            elif aux.ndim == 4:
                if aux.shape[1] != expected_channels:
                    raise ValueError(f"Auxiliary input '{name}' has {aux.shape[1]} channels, expected {expected_channels}.")
                aux = F.interpolate(aux, size=spatial_shape, mode="bilinear", align_corners=False)
            else:
                raise ValueError(f"Auxiliary input '{name}' must have shape (B,N,D) or (B,C,H,W), got {tuple(aux.shape)}.")
            aux_channels.append(aux)
        return aux_channels

    def forward(self, x: torch.Tensor, x_auxiliary: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        input_shape = (x.shape[-2], x.shape[-1])
        if self.use_coordconv:
            x = append_coord_channels(x)
        aux_channels = [aux.to(device=x.device, dtype=x.dtype) for aux in self._auxiliary_channels(x_auxiliary, input_shape)]
        if aux_channels:
            x = torch.cat([x, *aux_channels], dim=1)

        out = self.model(x)
        if out.shape[-2:] != input_shape:
            out = F.interpolate(out, size=input_shape, mode="bilinear", align_corners=False)
        return out
