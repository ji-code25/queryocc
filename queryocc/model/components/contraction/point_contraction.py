import logging
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

from torch import Tensor, autograd, nn

from queryocc.common.types import PointCloudRange
from queryocc.config.base_config import InstantiateConfig
from queryocc.model.components.contraction.contraction_encoders import (
    ProjectionMipNerf360,
    ProjectionMLP,
    ProjectionMLPHybrid,
    ProjectionMLPScaling,
    ProjectionTanh,
)

logger = logging.getLogger(__name__)


@dataclass
class PointContractionConfig(InstantiateConfig):
    """Configuration for Point Contraction settings."""

    _target: type = field(default_factory=lambda: PointContraction)

    dimensions: int = 2
    """The dimensions of the point contraction."""

    encoder: Literal["displacement", "scaling", "hybrid", "tanh", "mipnerf360"] = "mipnerf360"
    """The encoder to use for the point contraction."""

    hidden_dims: Tuple[int, ...] = (64, 32)
    """The hidden features of the encoder."""

    tanh_scaling_factor: float = 1.0
    """The scaling factor for the tanh encoder."""

    high_resolution_range: PointCloudRange = (-40, -40, 0, 40, 40, 3)
    """The high resolution range for the mipnerf360 encoder."""


class PointContraction(nn.Module):
    """Module for Point Contraction functionality."""

    def __init__(
        self,
        config: PointContractionConfig,
        point_cloud_range: PointCloudRange,
    ):
        super().__init__()
        self.config = config
        if self.config.encoder == "displacement":
            self.encoder = ProjectionMLP(
                point_dimensions=self.config.dimensions,
                hidden_features=self.config.hidden_dims,
            )
        elif self.config.encoder == "scaling":
            self.encoder = ProjectionMLPScaling(
                point_dimensions=self.config.dimensions,
                hidden_features=self.config.hidden_dims,
            )
        elif self.config.encoder == "hybrid":
            self.encoder = ProjectionMLPHybrid(
                point_dimensions=self.config.dimensions,
                hidden_features=self.config.hidden_dims,
            )
        elif self.config.encoder == "tanh":
            self.encoder = ProjectionTanh(
                scaling_factor=self.config.tanh_scaling_factor,
            )
        elif self.config.encoder == "mipnerf360":
            self.encoder = ProjectionMipNerf360(
                point_cloud_range=point_cloud_range,
                high_resolution_range=self.config.high_resolution_range,
            )
        else:
            raise ValueError(f"Unknown encoder: {self.config.encoder}")

    def _normalize_xy(self, xy: Tensor, point_cloud_range: PointCloudRange) -> Tensor:
        lower, upper = point_cloud_range[: self.config.dimensions], point_cloud_range[3 : 3 + self.config.dimensions]
        xy = 2 * (xy - lower) / (upper - lower) - 1
        return xy

    def _denormalize_xy(self, xy: Tensor, point_cloud_range: PointCloudRange) -> Tensor:
        lower, upper = point_cloud_range[: self.config.dimensions], point_cloud_range[3 : 3 + self.config.dimensions]
        xy = (xy + 1) / 2 * (upper - lower) + lower
        return xy

    def forward(self, xy: Tensor, point_cloud_range: Optional[PointCloudRange] = None) -> Tensor:
        assert xy.shape[-1] == self.config.dimensions, (
            f"Expected {self.config.dimensions} dimensions, got {xy.shape[-1]}"
        )

        if point_cloud_range is not None:
            xy = self._normalize_xy(xy, point_cloud_range)

        xy = self.encoder(xy)

        if point_cloud_range is not None:
            xy = self._denormalize_xy(xy, point_cloud_range)

        return xy


class VoxelizeSTE(autograd.Function):
    @staticmethod
    def forward(ctx, input_float):
        return input_float.long()

    @staticmethod
    def backward(ctx, grad_output_int):
        return grad_output_int.float()
