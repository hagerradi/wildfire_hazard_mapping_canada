import pytest
import torch

from src.models.baselines import UNet


@pytest.mark.parametrize("input_channels,num_classes", [(1, 1), (3, 2), (36, 1)])
def test_unet_output_shape(input_channels, num_classes):
    model = UNet(input_channels=input_channels, num_classes=num_classes)
    x = torch.randn(2, input_channels, 128, 128)
    out = model(x)
    assert out.shape == (2, num_classes, 128, 128)


def test_unet_forward_runs():
    model = UNet(input_channels=1, num_classes=1)
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 1, 64, 64)


def test_unet_skip_connections_toggle():
    x = torch.randn(1, 1, 64, 64)
    model_skip = UNet(input_channels=1, num_classes=1, use_skip_connections=True)
    model_no_skip = UNet(input_channels=1, num_classes=1, use_skip_connections=False)
    out_skip = model_skip(x)
    out_no_skip = model_no_skip(x)
    assert out_skip.shape == out_no_skip.shape


def test_unet_custom_hidden_features():
    model = UNet(input_channels=1, num_classes=1, hidden_features=[8, 16, 32])
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)
