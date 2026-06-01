import pytest
import torch

from src.models.encoders import BaselineEncoder, GlobalContextGridEncoder, GlobalContextMultiScaleEncoder, append_coord_channels
from src.models.unet import MultiSourceUNet


def test_global_context_encoder_matches_bottleneck_without_adaptive_pool():
    encoder = GlobalContextGridEncoder(in_channels=3, hidden_dims=[4, 8, 16], embed_dim=5, output_size=16)
    x = torch.randn(2, 3, 128, 128, requires_grad=True)

    y = encoder(x)
    y.mean().backward()

    assert y.shape == (2, 5, 16, 16)
    assert x.grad is not None
    assert not any(isinstance(module, torch.nn.AdaptiveAvgPool2d) for module in encoder.modules())


def test_global_context_encoder_rejects_non_integer_pool_to_bottleneck():
    encoder = GlobalContextGridEncoder(in_channels=3, hidden_dims=[4], embed_dim=5, output_size=7)
    x = torch.randn(2, 3, 17, 17)

    with pytest.raises(ValueError, match="must be divisible"):
        encoder(x)


def test_append_coord_channels_uses_patch_local_range():
    x = torch.zeros(2, 3, 4, 5)

    y = append_coord_channels(x)

    assert y.shape == (2, 5, 4, 5)
    torch.testing.assert_close(y[:, 3, :, 0], torch.linspace(-1.0, 1.0, 4).expand(2, 4))
    torch.testing.assert_close(y[:, 4, 0, :], torch.linspace(-1.0, 1.0, 5).expand(2, 5))


def test_baseline_encoder_coordconv_preserves_feature_shapes():
    encoder = BaselineEncoder(in_channels=3, hidden_features=[4, 8], use_coordconv=True)
    x = torch.randn(2, 3, 32, 32)

    y, skips = encoder(x)

    assert y.shape == (2, 8, 8, 8)
    assert [skip.shape for skip in skips] == [(2, 4, 32, 32), (2, 8, 16, 16)]


def test_global_context_multiscale_encoder_matches_unet_scales():
    encoder = GlobalContextMultiScaleEncoder(in_channels=3, hidden_features=[4, 8], bottleneck_channels=8, hidden_dim=4)
    x = torch.randn(2, 3, 64, 64, requires_grad=True)

    skips, bottleneck = encoder(x, skip_shapes=[(32, 32), (16, 16)], bottleneck_shape=(8, 8))
    loss = bottleneck.mean() + sum(skip.mean() for skip in skips)
    loss.backward()

    assert [skip.shape for skip in skips] == [(2, 4, 32, 32), (2, 8, 16, 16)]
    assert bottleneck.shape == (2, 8, 8, 8)
    assert x.grad is not None


def test_multisource_unet_supports_multiscale_context_and_coordconv():
    model = MultiSourceUNet(
        input_channels=3,
        num_classes=1,
        hidden_features=[4, 8],
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_input_dims={"weather": 2, "fire_size": 1, "global_context_grid": 3},
        auxiliary_hidden_dims={"weather": [4], "fire_size": [4], "global_context_grid": [4]},
        auxiliary_embed_dims={"weather": 4, "fire_size": 2, "global_context_grid": 4},
        auxiliary_feature_encoder_poolings={"weather": "mean", "fire_size": "mean"},
        use_coordconv=True,
        use_multiscale_global_context=True,
    )

    y = model(
        torch.randn(2, 3, 32, 32),
        {
            "weather": torch.randn(2, 5, 2),
            "fire_size": torch.randn(2, 5, 1),
            "global_context_grid": torch.randn(2, 3, 32, 32),
        },
    )

    assert y.shape == (2, 1, 32, 32)
