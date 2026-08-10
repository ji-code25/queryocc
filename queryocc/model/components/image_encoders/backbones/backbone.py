import torch
from torch import nn
from torch.profiler import record_function

from queryocc.model.components.image_encoders.backbones.convnext import ConvNeXtEncoderBase
from queryocc.model.components.image_encoders.backbones.necks import SimpleFPN, SimpleUp
from queryocc.model.components.image_encoders.backbones.resnet import Resnet18, Resnet34, Resnet50, Resnet101
from queryocc.model.components.image_encoders.common import IMAGENET_MEAN, IMAGENET_STD

# Exact-name backbones. The convnext_* prefix family (-> ConvNeXtEncoderBase) is handled in
# build_backbone. Every backbone exposes `out_channels`
# (its per-stage feature dims), which sizes the neck — there is no separate channel table to
# keep in sync.
RESNET_BACKBONES = {
    "resnet18": Resnet18,
    "resnet34": Resnet34,
    "resnet50": Resnet50,
    "resnet101": Resnet101,
}

NECKS = {
    "simpleup": SimpleUp,
    "simplefpn": SimpleFPN,
}


def build_backbone(
    backbone_type: str,
    gradient_checkpointing: bool = False,
    swap_batch_norm_to_instance_norm: bool = False,
    pretrained: bool = True,
    **kwargs,
) -> nn.Module:
    """``pretrained=False`` builds the backbone without its published initialization
    (DINOv3 for convnext_*, ImageNet for resnet*). Set it when a full QueryOcc checkpoint
    is about to overwrite these weights anyway — it drops both the gated DINOv3 file
    requirement and the torchvision download from the evaluation path."""
    name = backbone_type.lower()
    if name in RESNET_BACKBONES:
        return RESNET_BACKBONES[name](
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            pretrained=pretrained,
            **kwargs,
        )
    if name.startswith("convnext"):
        arch = name.split("_")[1]
        return ConvNeXtEncoderBase(arch, gradient_checkpointing=gradient_checkpointing, pretrained=pretrained)
    raise ValueError(
        f"Unknown backbone type {backbone_type}. Must be one of {sorted(RESNET_BACKBONES)} or start with 'convnext'."
    )


class BackboneEncoder(nn.Module):
    def __init__(
        self,
        backbone_type="resnet50",
        neck_type="simpleup",
        feature_dim=128,
        encode_time=False,
        time_encoding_max_freq=6.0,
        gradient_checkpointing=False,
        swap_batch_norm_to_instance_norm=False,
        encode_time_fusion_type="sum",
        pretrained=True,
        **kwargs,
    ):
        super().__init__()
        self.backbone = build_backbone(
            backbone_type,
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            pretrained=pretrained,
            **kwargs,
        )

        if neck_type.lower() not in NECKS:
            raise ValueError(f"Unknown neck type {neck_type}. Must be one of {sorted(NECKS)}")
        self.neck = NECKS[neck_type.lower()](
            feature_dim,
            in_channels=self.backbone.out_channels,
            encode_time=encode_time,
            encode_time_fusion_type=encode_time_fusion_type,
            time_encoding_max_freq=time_encoding_max_freq,
        )

        for name, value in (("_resnet_mean", IMAGENET_MEAN), ("_resnet_std", IMAGENET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)

    def _normalize(self, x):
        return (x - self._resnet_mean) / self._resnet_std

    @record_function("BackboneEncoder.forward")
    def forward(self, x, ts, maybe_checkpoint=True) -> torch.Tensor:
        x = self._normalize(x)
        x = self.backbone(x, maybe_checkpoint=maybe_checkpoint)
        x = self.neck(x, ts)
        return x
