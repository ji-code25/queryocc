"""Image encoders: camera input -> scene features for QueryOccNet.

One encoder family, selected by ``NetConfig.image_encoder_name`` (see net.py):

- ``cerveza/`` — the BEV lift pipeline: backbone -> neck -> LSS depth-splat lift ->
  temporal fusion -> BEV grid. Config ``NetConfig.cerveza_encoder``.

Shared infrastructure: ``backbones/`` (backbone registry + necks), ``lifters/`` (LSS lift,
depth net, lidar targets, simple-BEV), ``geometry.py`` (analytic matrix inverses — never
torch.inverse on GPU), ``common.py``.
"""

from queryocc.model.components.image_encoders.backbones.backbone import BackboneEncoder, build_backbone
from queryocc.model.components.image_encoders.cerveza.encoder import CervezaImageEncoder, CervezaImageEncoderConfig
from queryocc.model.components.image_encoders.lifters.lss import LSSPro, LSSProConfig

__all__ = [
    "BackboneEncoder",
    "CervezaImageEncoder",
    "CervezaImageEncoderConfig",
    "LSSPro",
    "LSSProConfig",
    "build_backbone",
]
