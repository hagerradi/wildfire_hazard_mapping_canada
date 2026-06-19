import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.unet import BaselineUNet, MultiSourceUNet


def test_auto_factory_preserves_baseline_unet_path():
    config = ModelConfig(hidden_features=[8, 16], input_branches=["spatial"])

    model = build_model(model_config=config, spatial_input_channels=3)

    assert resolve_model_architecture(config) == "baseline_unet"
    assert isinstance(model, BaselineUNet)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_auto_factory_preserves_multi_source_unet_path():
    config = ModelConfig(
        hidden_features=[8, 16],
        input_branches=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"tabular_weather": [8]},
        auxiliary_embed_dims={"tabular_weather": 8},
    )

    model = build_model(model_config=config, spatial_input_channels=3, auxiliary_input_dims={"tabular_weather": 5})

    assert resolve_model_architecture(config) == "multi_source_unet"
    assert isinstance(model, MultiSourceUNet)
    out = model(torch.randn(2, 3, 32, 32), {"tabular_weather": torch.randn(2, 4, 5)})
    assert out.shape == (2, 1, 32, 32)
