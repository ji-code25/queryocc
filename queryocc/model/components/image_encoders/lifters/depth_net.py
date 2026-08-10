"""BEVDepth-style depth network for the LSS lifter (vendored components).

DepthNet + its ASPP/SE/BasicBlock building blocks follow the BEVDepth recipe
(https://github.com/Megvii-BaseDetection/BEVDepth): camera-parameter-conditioned SE
gating over a shared trunk, an ASPP context stage, and a 1x1 head emitting
[depth logits; context features] concatenated on the channel dim.
"""

from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.profiler import record_function

from queryocc.model.components.checkpointing import checkpoint


class Mlp(nn.Module):
    """Two-layer MLP (from facebookresearch/vggt, Apache 2.0; originally timm)."""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DepthNet(nn.Module):
    def __init__(
        self,
        in_channels,
        mid_channels,
        context_channels,
        depth_channels,
        gradient_checkpointing: bool = False,
        swap_batch_norm_to_instance_norm: bool = False,
    ):
        super(DepthNet, self).__init__()

        self.gradient_checkpointing = gradient_checkpointing

        if swap_batch_norm_to_instance_norm:
            Norm2D = nn.InstanceNorm2d
            Norm1D = nn.InstanceNorm1d
        else:
            Norm2D = nn.BatchNorm2d
            Norm1D = nn.BatchNorm1d

        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1),
            Norm2D(mid_channels),
            nn.ReLU(inplace=True),
        )

        self.context_conv = nn.Conv2d(mid_channels, context_channels, kernel_size=1, stride=1, padding=0)
        self.context_mlp = Mlp(16, mid_channels, mid_channels, act_layer=nn.ReLU)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware

        self.bn = Norm1D(16)
        self.depth_mlp = Mlp(16, mid_channels, mid_channels, act_layer=nn.ReLU)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware

        depth_conv_list = [
            BasicBlock(mid_channels, mid_channels, Norm2D),
            BasicBlock(mid_channels, mid_channels, Norm2D),
            BasicBlock(mid_channels, mid_channels, Norm2D),
            ASPP(mid_channels, mid_channels, Norm2D),
            nn.Conv2d(mid_channels, depth_channels, kernel_size=1, stride=1, padding=0),
        ]

        self.depth_conv = nn.Sequential(*depth_conv_list)

    @record_function("DepthNet.forward")
    def forward(self, x, mlp_input):
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)

        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)

        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)

        context = self.context_conv(context)

        if self.gradient_checkpointing:
            depth = checkpoint(self.depth_conv, depth, use_reentrant=False)
        else:
            depth = self.depth_conv(depth)

        return torch.cat([depth, context], dim=1)


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, Norm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(
            inplanes,
            planes,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = Norm(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    @record_function("ASPPModule.forward")
    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, Norm=nn.BatchNorm2d):
        super(ASPP, self).__init__()

        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(
            inplanes,
            mid_channels,
            1,
            padding=0,
            dilation=dilations[0],
            Norm=Norm,
        )
        self.aspp2 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[1],
            dilation=dilations[1],
            Norm=Norm,
        )
        self.aspp3 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[2],
            dilation=dilations[2],
            Norm=Norm,
        )
        self.aspp4 = _ASPPModule(
            inplanes,
            mid_channels,
            3,
            padding=dilations[3],
            dilation=dilations[3],
            Norm=Norm,
        )

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            nn.BatchNorm2d(mid_channels) if Norm == nn.BatchNorm2d else nn.Identity(),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5), inplanes, 1, bias=False)
        self.bn1 = Norm(inplanes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    @record_function("ASPP.forward")
    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode="bilinear", align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, Norm):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = Norm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = Norm(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
