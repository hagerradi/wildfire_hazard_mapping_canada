import random

import torch
import torchvision.transforms.functional as F


class RandomFlip:
    """Performs either a horizontal or vertical flip (equal chance)."""

    def __call__(self, x, y, mask):
        if random.random() > 0.5:
            x, y, mask = F.hflip(x), F.hflip(y), F.hflip(mask)
        else:
            x, y, mask = F.vflip(x), F.vflip(y), F.vflip(mask)
        return x, y, mask


class RandomRotate90:
    """Performs a random 90, 180, or 270 degree rotation."""

    def __call__(self, x, y, mask):
        k = random.randint(1, 3)
        x = torch.rot90(x, k, dims=[1, 2])
        y = torch.rot90(y, k, dims=[1, 2])
        mask = torch.rot90(mask, k, dims=[1, 2])
        return x, y, mask


class Compose:
    """Runs the selected transforms with a global prob."""

    def __init__(self, transforms_list, prob):
        self.transforms = transforms_list
        self.prob = prob

    def __call__(self, x, y, mask):
        for t in self.transforms:
            if random.random() < self.prob:
                x, y, mask = t(x, y, mask)
        return x, y, mask


def setup_augmentations(config_data):
    """Utils. to get the list of transforms from config."""
    if not config_data.transforms_list:
        return None

    mapping = {
        "random_flip": RandomFlip(),
        "random_rotate": RandomRotate90(),
    }

    selected = [mapping[name] for name in config_data.transforms_list if name in mapping]

    if not selected:
        return None

    return Compose(selected, prob=config_data.augmentation_prob)
