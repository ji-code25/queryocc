"""Small utilities shared across the image encoder package."""

from typing import Dict, Optional

import torch
from torch import Tensor

# ImageNet statistics used to normalize RGB inputs for every pretrained backbone
# (torchvision ResNets, DINOv2/v3, ConvNeXt all train with these).
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def stack_dict_of_tensors(dict_of_tensors: Optional[Dict[str, Tensor]]) -> Optional[Tensor]:
    """Stack a per-camera dict {name: (B, ...)} into one (N_cams, B, ...) tensor.

    Returns None for None/empty input so optional streams (depth maps, distortion) pass through.
    """
    if dict_of_tensors is None or len(dict_of_tensors) == 0:
        return None
    return torch.stack(list(dict_of_tensors.values()))
