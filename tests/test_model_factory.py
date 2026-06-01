import pytest
import torch

from src.config import ModelConfig
from src.models.factory import build_model, resolve_model_architecture
from src.models.smp import SMPDensePredictionModel
from src.models.unet import BaselineUNet, MultiSourceUNet


def test_auto_factory_preserves_baseline_unet_path():
    config = ModelConfig(hidden_features=[8, 16], input_feature_list=["spatial"])

    model = build_model(model_config=config, spatial_input_channels=3)

    assert resolve_model_architecture(config) == "baseline_unet"
    assert isinstance(model, BaselineUNet)
    out = model(torch.randn(2, 3, 32, 32))
    assert out.shape == (2, 1, 32, 32)


def test_auto_factory_preserves_multi_source_unet_path():
    config = ModelConfig(
        hidden_features=[8, 16],
        input_feature_list=["spatial", "auxiliary"],
        auxiliary_hidden_dims={"weather": [8]},
        auxiliary_embed_dims={"weather": 8},
    )

    model = build_model(model_config=config, spatial_input_channels=3, auxiliary_input_dims={"weather": 5})

    assert resolve_model_architecture(config) == "multi_source_unet"
    assert isinstance(model, MultiSourceUNet)
    out = model(torch.randn(2, 3, 32, 32), {"weather": torch.randn(2, 4, 5)})
    assert out.shape == (2, 1, 32, 32)


def test_smp_factory_forward_shape_with_coordconv():
    config = ModelConfig(
        architecture="smp",
        input_feature_list=["spatial"],
        num_classes=2,
        use_coordconv=True,
        smp_architecture="Unet",
        smp_encoder_name="resnet18",
        smp_encoder_weights=None,
    )

    model = build_model(model_config=config, spatial_input_channels=3)

    assert isinstance(model, SMPDensePredictionModel)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 2, 64, 64)


def test_smp_factory_supports_auxiliary_features_as_channels():
    config = ModelConfig(
        architecture="smp",
        input_feature_list=["spatial", "auxiliary"],
        use_coordconv=True,
        smp_architecture="Unet",
        smp_encoder_name="resnet18",
        smp_encoder_weights=None,
        auxiliary_feature_encoder_poolings={"weather": "max"},
    )

    model = build_model(model_config=config, spatial_input_channels=3, auxiliary_input_dims={"weather": 5})

    out = model(torch.randn(2, 3, 64, 64), {"weather": torch.randn(2, 4, 5)})
    assert out.shape == (2, 1, 64, 64)


def test_smp_factory_rejects_missing_requested_auxiliary_dimensions():
    config = ModelConfig(architecture="smp", input_feature_list=["spatial", "auxiliary"], smp_encoder_weights=None)

    with pytest.raises(ValueError, match="auxiliary dimensions"):
        build_model(model_config=config, spatial_input_channels=3)
