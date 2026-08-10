import torch
from torch import nn
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    resnet18,
    resnet34,
    resnet50,
    resnet101,
)

from queryocc.model.components.checkpointing import checkpoint


class ResnetEncoderBase(nn.Module):
    out_channels: list[int]  # per-stage dims of the [layer2, layer3, layer4] maps forward returns

    def __init__(self, gradient_checkpointing=False, swap_batch_norm_to_instance_norm=False, **kwargs):
        super().__init__()
        self.gradient_checkpointing = gradient_checkpointing
        self.swap_batch_norm_to_instance_norm = swap_batch_norm_to_instance_norm
        self.backbone = None
        self.layer2 = None
        self.layer3 = None
        self.layer4 = None

    def _swap_batch_norm_to_instance_norm(self):
        # Replace via setattr on the parent: reassigning the loop variable (the previous
        # implementation) never touches the model, silently leaving BatchNorm in place.
        if self.swap_batch_norm_to_instance_norm:
            for module in self.modules():
                for child_name, child in module.named_children():
                    if isinstance(child, nn.BatchNorm2d):
                        setattr(module, child_name, nn.InstanceNorm2d(child.num_features, affine=True))

    def maybe_checkpoint(self, fn, *args, **kwargs):
        """Apply gradient checkpointing if enabled."""
        return checkpoint(fn, *args, **kwargs) if self.gradient_checkpointing else fn(*args)

    def forward(self, x, maybe_checkpoint=True) -> list[torch.Tensor]:
        if self.gradient_checkpointing and maybe_checkpoint:
            x1 = self.backbone(x)  # type: ignore
            x2 = self.maybe_checkpoint(self.layer2, x1, use_reentrant=False)
            x3 = self.maybe_checkpoint(self.layer3, x2, use_reentrant=False)
            x4 = self.maybe_checkpoint(self.layer4, x3, use_reentrant=False)
        else:
            x1 = self.backbone(x)  # type: ignore
            x2 = self.layer2(x1)  # type: ignore
            x3 = self.layer3(x2)  # type: ignore
            x4 = self.layer4(x3)  # type: ignore

        return [x2, x3, x4]


class Resnet18(ResnetEncoderBase):
    out_channels = [128, 256, 512]

    def __init__(self, gradient_checkpointing=False, swap_batch_norm_to_instance_norm=False, pretrained=True, **kwargs):
        super().__init__(
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            **kwargs,
        )
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        if self.swap_batch_norm_to_instance_norm:
            self._swap_batch_norm_to_instance_norm()


class Resnet34(ResnetEncoderBase):
    out_channels = [128, 256, 512]

    def __init__(self, gradient_checkpointing=False, swap_batch_norm_to_instance_norm=False, pretrained=True, **kwargs):
        super().__init__(
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            **kwargs,
        )
        resnet = resnet34(weights=ResNet34_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.swap_batch_norm_to_instance_norm:
            self._swap_batch_norm_to_instance_norm()


class Resnet50(ResnetEncoderBase):
    out_channels = [512, 1024, 2048]

    def __init__(self, gradient_checkpointing=False, swap_batch_norm_to_instance_norm=False, pretrained=True, **kwargs):
        super().__init__(
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            **kwargs,
        )
        resnet = resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.swap_batch_norm_to_instance_norm:
            self._swap_batch_norm_to_instance_norm()


class Resnet101(ResnetEncoderBase):
    out_channels = [512, 1024, 2048]

    def __init__(self, gradient_checkpointing=False, swap_batch_norm_to_instance_norm=False, pretrained=True, **kwargs):
        super().__init__(
            gradient_checkpointing=gradient_checkpointing,
            swap_batch_norm_to_instance_norm=swap_batch_norm_to_instance_norm,
            **kwargs,
        )
        resnet = resnet101(weights=ResNet101_Weights.DEFAULT if pretrained else None)
        self.backbone = nn.Sequential(*list(resnet.children())[:5])
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        if self.swap_batch_norm_to_instance_norm:
            self._swap_batch_norm_to_instance_norm()
