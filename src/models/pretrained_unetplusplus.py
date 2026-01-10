import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class PretrainedUNetPlusPlus(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        decoder_interpolation: str = "bilinear",
        **kwargs,
    ):
        """
        Wrapper for a pretrained SMP UNet++.

        Args:
            input_channels (int): number of input channels (features)
            num_classes (int): number of output channels
            encoder_name (str): type of encoder to use (see SMP doc. for more)
            encoder_weights (str | None): pretrained weights to use (see SMP doc. for more)
            decoder_interpolation (str): type of interpolation to use (e.g. "nearest", "bilinear")
        """
        super().__init__()

        self.model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=input_channels,
            classes=num_classes,
            decoder_interpolation=decoder_interpolation,
            **kwargs,
        )

        self.encoder_name = encoder_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    model = PretrainedUNetPlusPlus(input_channels=36, num_classes=1)
    print(f"Model created with encoder: {model.encoder_name}")
    x = torch.randn(2, 36, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
