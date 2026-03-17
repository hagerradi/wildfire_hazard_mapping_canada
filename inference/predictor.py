"""
Pure inference class for wildfire burn risk prediction.
Wraps the PyTorch model and handles tensor-in, tensor-out operations.
"""

import logging
from pathlib import Path
from typing import Any

import torch

from src.models.unet import BaselineUNet, MultiSourceUNet

logger = logging.getLogger(__name__)

class BurnRiskPredictor:
    """
    Pure inference class for wildfire burn risk prediction.
    Wraps the PyTorch model and handles tensor-in, tensor-out operations.
    
    This class is focusses only on model operations.
    
    Example:
        >>> predictor = BurnRiskPredictor.from_checkpoint(
        ...     checkpoint_path="models/best.pth",
        ...     spatial_channels=10,
        ...     auxiliary_input_dims={"weather": 7, "fire_size": 5}
        ... )
        >>> predictions = predictor(spatial_batch, auxiliary_batch)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        device: str | torch.device,
        config: dict[str, Any] | None = None,
    ):
        """
        Initialize with an already-built model.
        
        For typical usage, prefer the `from_checkpoint` classmethod.
        
        Args:
            model: A PyTorch model (BaselineUNet or MultiSourceUNet).
            device: Device the model is on.
            config: Optional config dict for reference.
        """
        self.model = model
        self.device = device
        self.config = config
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        spatial_channels: int,
        auxiliary_input_dims: dict[str, int] | None = None,
        device: str | torch.device | None = None,
    ) -> "BurnRiskPredictor":
        """
        Create a predictor from a saved checkpoint file.
        
        Args:
            checkpoint_path: Path to the trained model checkpoint (.pth file). Must contain 'model_state' and 'config'.
            spatial_channels: Number of input channels for spatial data.
            auxiliary_input_dims: Dict mapping auxiliary source names to their dimensions.
            device: Device to run inference on. If None, auto-detects GPU/CPU.
            
        Returns:
            BurnRiskPredictor instance ready for inference.
        """
        checkpoint_path = Path(checkpoint_path)
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        logger.info(f"Loading checkpoint from {checkpoint_path} on {device}")
        
        config = checkpoint["config"]
        model_config = config["model"]

        # Build model architecture
        model = cls._build_model(
            model_config=model_config,
            spatial_channels=spatial_channels,
            auxiliary_input_dims=auxiliary_input_dims,
        )

        # Load weights
        model.load_state_dict(checkpoint["model_state"])
        model.to(device)
        model.eval()

        logger.info(f"Model loaded successfully on {device}")
        return cls(model=model, device=device, config=config)

    @staticmethod
    def _build_model(
        model_config: dict,
        spatial_channels: int,
        auxiliary_input_dims: dict[str, int],
    ) -> torch.nn.Module:
        """Instantiate the model architecture based on config."""
        input_feature_list = model_config["input_feature_list"]
        use_auxiliary = "auxiliary" in input_feature_list

        if use_auxiliary:
            return MultiSourceUNet(
                input_channels=spatial_channels,
                num_classes=model_config["num_classes"],
                hidden_features=model_config["hidden_features"],
                input_feature_list=input_feature_list,
                use_skip_connections=model_config["use_skip_connections"],
                use_transpose_conv=model_config["use_transpose_conv"],
                use_activation_after_upsampling=model_config["use_activation_after_upsampling"],
                auxiliary_input_dims=auxiliary_input_dims,
                auxiliary_hidden_dims=model_config["auxiliary_hidden_dims"],
                auxiliary_embed_dims=model_config["auxiliary_embed_dims"],
                auxiliary_feature_encoder_poolings=model_config["auxiliary_feature_encoder_poolings"],
            )
        else:
            return BaselineUNet(
                input_channels=spatial_channels,
                num_classes=model_config["num_classes"],
                hidden_features=model_config["hidden_features"],
                input_feature_list=input_feature_list,
                use_skip_connections=model_config["use_skip_connections"],
                use_transpose_conv=model_config["use_transpose_conv"],
                use_activation_after_upsampling=model_config["use_activation_after_upsampling"],
            )

    @torch.no_grad()
    def predict_batch(
        self,
        spatial_inputs: torch.Tensor,
        auxiliary_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """
        Run a single batch through the model.
        
        Args:
            spatial_inputs: Spatial grid tensor of shape (B, C, H, W).
            auxiliary_inputs: Optional dict of auxiliary tensors.
            
        Returns:
            Predictions tensor of shape (B, num_classes, H, W) on CPU.
        """
        spatial_inputs = spatial_inputs.to(self.device)

        if auxiliary_inputs:
            # Remove 'grid' from auxiliary inputs if present, as it's already passed as spatial_inputs
            auxiliary_inputs = {
                k: v.to(self.device) for k, v in auxiliary_inputs.items() if k != "grid"
            }

        predictions = self.model(spatial_inputs, auxiliary_inputs if auxiliary_inputs else None)
        
        # Move predictions to CPU before returning
        return predictions.cpu()

    def __call__(
        self,
        spatial_inputs: torch.Tensor,
        auxiliary_inputs: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Shorthand for predict_batch."""
        return self.predict_batch(spatial_inputs, auxiliary_inputs)
