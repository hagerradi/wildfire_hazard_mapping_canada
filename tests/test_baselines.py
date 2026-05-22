import pytest
import torch

from src.models.encoders import append_coord_channels
from src.models.unet import BaselineUNet, MultiSourceUNet


@pytest.mark.parametrize(("input_channels", "num_classes"), [(1, 1), (3, 2), (36, 1)])
def test_unet_output_shape_and_forward_runs(input_channels, num_classes):
    model = BaselineUNet(input_channels=input_channels, num_classes=num_classes)
    x = torch.randn(2, input_channels, 128, 128)
    out = model(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, num_classes, 128, 128)


def test_unet_skip_connections_toggle():
    x = torch.randn(1, 1, 64, 64)
    model_skip = BaselineUNet(input_channels=1, num_classes=1, use_skip_connections=True)
    model_no_skip = BaselineUNet(input_channels=1, num_classes=1, use_skip_connections=False)
    out_skip = model_skip(x)
    out_no_skip = model_no_skip(x)
    assert out_skip.shape == out_no_skip.shape


def test_unet_custom_hidden_features():
    model = BaselineUNet(input_channels=1, num_classes=1, hidden_features=[8, 16, 32])
    x = torch.randn(1, 1, 64, 64)
    out = model(x)
    assert out.shape == (1, 1, 64, 64)


def test_append_coord_channels_uses_patch_local_coordinates():
    x = torch.zeros(1, 1, 2, 3)
    out = append_coord_channels(x)

    assert out.shape == (1, 3, 2, 3)
    torch.testing.assert_close(out[0, 1], torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]))
    torch.testing.assert_close(out[0, 2], torch.tensor([[-1.0, 0.0, 1.0], [-1.0, 0.0, 1.0]]))


def test_unet_coordconv_forward_shape():
    model = BaselineUNet(input_channels=3, num_classes=1, hidden_features=[8, 16], use_coordconv=True)
    x = torch.randn(2, 3, 64, 64)

    out = model(x)

    assert out.shape == (2, 1, 64, 64)


def test_multisource_unet_coordconv_forward_shape():
    model = MultiSourceUNet(
        input_channels=3,
        num_classes=1,
        hidden_features=[8, 16],
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_input_dims={"weather": 5},
        auxiliary_hidden_dims={"weather": [8]},
        auxiliary_embed_dims={"weather": 4},
        auxiliary_feature_encoder_poolings={"weather": "max"},
        use_coordconv=True,
    )
    x = torch.randn(2, 3, 64, 64)
    x_auxiliary = {"weather": torch.randn(2, 4, 5)}

    out = model(x, x_auxiliary=x_auxiliary)

    assert out.shape == (2, 1, 64, 64)
