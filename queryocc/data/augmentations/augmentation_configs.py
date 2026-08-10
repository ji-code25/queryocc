from dataclasses import dataclass
from typing import Tuple


@dataclass
class AugmentationConfig:
    # grid mask
    grid_mask: bool = True
    grid_mask_prob: float = 0.5
    grid_mask_rotate: float = 45
    grid_mask_offset: bool = False
    grid_mask_ratio: float = 0.80
    grid_mask_mode: int = 1
    grid_mask_use_h: bool = True
    grid_mask_use_w: bool = True
    # photometric distortion
    photometric_distortion_brightness: Tuple[float, float] = (0.875, 1.125)
    photometric_distortion_contrast: Tuple[float, float] = (0.5, 1.5)
    photometric_distortion_saturation: Tuple[float, float] = (0.5, 1.5)
    photometric_distortion_hue: Tuple[float, float] = (-0.05, 0.05)
