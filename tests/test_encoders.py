import torch

from src.models.encoders import BaselineEncoder, append_coord_channels


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
