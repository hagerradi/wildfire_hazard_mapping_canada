# utils for models

import torch.nn as nn


def get_nbr_model_parameters(model: nn.Module) -> tuple[int, int]:
    """
    Gets the total and trainable parameters of a torch model.

    Args:
        model (nn.Module): The torch model.

    Returns:
        tuple: (total_parameters, trainable_parameters)
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params
