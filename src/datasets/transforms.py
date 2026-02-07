import random

import torch
import torchvision.transforms.functional as F

from src.config import DataSourceConfig


class RandomFlip:
    """Performs either a horizontal or vertical flip (equal chance)."""

    def __call__(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if random.random() > 0.5:
            x, target, mask = F.hflip(x), F.hflip(target), F.hflip(mask)
        else:
            x, target, mask = F.vflip(x), F.vflip(target), F.vflip(mask)
        return x, target, mask


class RandomRotate90:
    """Performs a random 90, 180, or 270 degree rotation."""

    def __call__(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k = random.randint(1, 3)  # k is the number of 90deg rotations to do
        x = torch.rot90(x, k, dims=[1, 2])
        target = torch.rot90(target, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])
        return x, target, mask


class Compose:
    """Runs the selected transforms with a global prob."""

    def __init__(self, transforms_list: list, prob: float):
        self.transforms = transforms_list
        self.prob = prob

    def __call__(self, x: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for t in self.transforms:
            # we apply the prob. independently for each transform in the list
            if random.random() < self.prob:
                x, target, mask = t(x, target, mask)
        return x, target, mask


def setup_augmentations(config: DataSourceConfig):
    """Utils. to get the list of transforms from config."""
    transform = config.params.get("transform", [])
    augmentation_prob = config.params.get("augmentatation_prob", 0.0)
    if not transform:
        return None

    # we can add future transforms here
    mapping = {
        "random_flip": RandomFlip(),
        "random_rotate": RandomRotate90(),
    }

    # check if the config keys match the options
    valid_keys = set(mapping.keys())
    config_keys = set(config.transforms_list)
    unknown_keys = config_keys - valid_keys

    if unknown_keys:
        raise ValueError(f"Invalid transforms found in config: {unknown_keys}.\n" f"Allowed options are: {list(valid_keys)}")

    selected = [mapping[name] for name in config.transform]
    return Compose(selected, prob=augmentation_prob)


def get_transforms(config: DataSourceConfig):
    """Returns Compose of DataSource specific transforms"""
    transform = None
    if config.name == "grid":
        transform = setup_augmentations(config)
    else:
        return transform
