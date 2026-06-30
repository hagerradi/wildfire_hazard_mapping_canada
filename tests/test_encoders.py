import pytest
import torch

from src.models.encoders import BaselineEncoder, FuelCurveEncoder, append_coord_channels


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


def test_fuel_curve_encoder_registers_buffers_and_returns_expected_shape():
    encoder = FuelCurveEncoder(
        in_channels=18,
        hidden_dim=16,
        embed_dim=4,
        curve_mean=torch.zeros(18),
        curve_std=torch.ones(18),
    )
    x = torch.rand(2, 18, 8, 6)

    y = encoder(x)
    buffers = dict(encoder.named_buffers())

    assert y.shape == (2, 4, 8, 6)
    assert {"curve_mean", "curve_std", "isi_pos"}.issubset(buffers)
    assert encoder.curve_mean.shape == (1, 18, 1, 1)
    assert encoder.curve_std.shape == (1, 18, 1, 1)
    assert encoder.isi_pos.shape == (1, 18, 1, 1)


def test_fuel_curve_encoder_raises_on_invalid_input_shape_or_channels():
    encoder = FuelCurveEncoder(
        in_channels=18,
        hidden_dim=16,
        embed_dim=4,
        curve_mean=torch.zeros(18),
        curve_std=torch.ones(18),
    )

    with pytest.raises(ValueError, match="Expected a 4D tensor"):
        encoder(torch.rand(2, 18, 8))

    with pytest.raises(ValueError, match="Expected 18 fuel curve channels"):
        encoder(torch.rand(2, 17, 8, 8))
