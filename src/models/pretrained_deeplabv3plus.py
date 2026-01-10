import segmentation_models_pytorch as smp
import torch
import torch.nn as nn


class PretrainedDeepLabV3Plus(nn.Module):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 1,
        encoder_name: str = "resnet34",
        encoder_weights: str | None = "imagenet",
        **kwargs,
    ):
        """
        Wrapper for a pretrained SMP DeepLabV3+.

        Args:
            input_channels (int): number of input channels (features)
            num_classes (int): number of output channels
            encoder_name (str): type of encoder to use (see SMP doc. for more)
            encoder_weights (str | None): pretrained weights to use (see SMP doc. for more)
        """
        super().__init__()

        self.model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=input_channels,
            classes=num_classes,
            **kwargs,
        )

        self.encoder_name = encoder_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


if __name__ == "__main__":
    model = PretrainedDeepLabV3Plus(input_channels=36, num_classes=1)
    print(f"Model created with encoder: {model.encoder_name}")
    x = torch.randn(2, 36, 256, 256)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
