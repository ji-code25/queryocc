import torch
import torch.nn.functional as F
from torch import nn

from queryocc.model.components.positional_encoder import CoordinateEncoder


class SimpleUp(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: list[int],
        encode_time=False,
        encode_time_fusion_type="sum",
        time_encoding_max_freq=6.0,
    ):
        super().__init__()

        if len(in_channels) > 1:
            self.upsample = UpsamplingConcat(out_channels * 2, out_channels)
        else:
            self.upsample = None

        self.reduce_dims = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(in_, out_channels, kernel_size=1, padding=0, bias=False),
                    nn.InstanceNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
                for in_ in in_channels
            ]
        )

        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        if encode_time:
            self.time_encoding = TimeEncoding(
                out_channels,
                fusion_type=encode_time_fusion_type,
                coordinate_dim=1,
                hidden_dim=128,
                num_freqs=16,
                max_freq=time_encoding_max_freq,
            )
        else:
            self.time_encoding = None

    def forward(self, x: list, ts: torch.Tensor) -> torch.Tensor:
        for i in range(len(x)):
            x[i] = self.reduce_dims[i](x[i])

        # upsample and concatenate, starting from the top
        if self.upsample is not None:
            for i in range(len(x) - 1, 0, -1):
                x[i - 1] = self.upsample(x[i], x[i - 1], i)

        # return the upsampled and concatenated features
        if self.time_encoding is not None and ts is not None:
            return self.depthwise(self.time_encoding(x[0], ts))
        else:
            return self.depthwise(x[0])


def match_shapes(x, y) -> torch.Tensor:
    if x.shape[-2:] != y.shape[-2:]:
        x = nn.functional.interpolate(x, size=y.shape[-2:], mode="bilinear", align_corners=False)
    return x


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels: int, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode="bilinear", align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x, i) -> torch.Tensor:
        x_to_upsample = self.upsample(x_to_upsample)

        x_to_upsample = match_shapes(x_to_upsample, x)

        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)

        return self.conv(x_to_upsample)


class SimpleFPN(nn.Module):
    def __init__(
        self,
        out_channels: int,
        in_channels: list[int],
        encode_time=False,
        encode_time_fusion_type="sum",
        time_encoding_max_freq=6.0,
    ):
        super().__init__()

        # Lateral connections
        self.lateral_convs = nn.ModuleList([nn.Conv2d(in_ch, out_channels, kernel_size=1) for in_ch in in_channels])

        # FPN convolutions
        self.fpn_convs = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.depthwise = nn.Conv2d(out_channels, out_channels, kernel_size=1, padding=0)

        if encode_time:
            self.time_encoding = TimeEncoding(
                out_channels,
                fusion_type=encode_time_fusion_type,
                coordinate_dim=1,
                hidden_dim=128,
                num_freqs=16,
                max_freq=time_encoding_max_freq,
            )
        else:
            self.time_encoding = None

    def forward(self, x: list, ts: torch.Tensor) -> torch.Tensor:
        # Build laterals
        laterals = [l_conv(x[i]) for i, l_conv in enumerate(self.lateral_convs)]

        # Build top-down path
        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] += upsampled

        # Build outputs (only the highest resolution feature map)
        outs = self.fpn_convs(laterals[0])

        # Time encoding
        if self.time_encoding is not None and ts is not None:
            outs = self.time_encoding(outs, ts)

        # Depthwise convolution
        return self.depthwise(outs)


class TimeEncoding(nn.Module):
    def __init__(self, out_channels, fusion_type="sum", coordinate_dim=1, hidden_dim=128, num_freqs=16, max_freq=4.0):
        """
        Args:
            out_channels (int): output feature dimension of the encoding
            fusion_type (str): one of "sum", "concat", or "film"
            coordinate_dim (int): input coordinate dimension (usually 1 for time)
            hidden_dim (int): hidden dimension of CoordinateEncoder
            num_freqs (int): number of frequencies for Fourier features
        """
        super().__init__()
        assert fusion_type in ("sum", "concat", "film"), f"Invalid fusion_type: {fusion_type}"
        self.fusion_type = fusion_type
        self.coordinate_encoder = CoordinateEncoder(
            coordinate_dim=coordinate_dim,
            hidden_dim=hidden_dim,
            out_dim=out_channels,
            num_freqs=num_freqs,
            max_freq=max_freq,
        )

        if fusion_type == "film":
            self.gamma_layer = nn.Linear(out_channels, out_channels)
            self.beta_layer = nn.Linear(out_channels, out_channels)
        elif fusion_type == "concat":
            # To fuse after concatenation, we need a conv to reduce channels back to out_channels
            self.proj_conv = nn.Conv2d(out_channels * 2, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        """
        Args:
            x: feature map tensor of shape (B, C, H, W)
            ts: Tensor of shape (B,) or (B, 1) representing time(s)
        Returns:
            fused feature map of shape (B, C, H, W)
        """
        # reshape handles both (B,) and (B, 1); squeeze(1) keeps the batch dim when B == 1
        time_emb = self.coordinate_encoder(ts.reshape(-1, 1, 1)).squeeze(1)  # (B, C)

        if self.fusion_type == "film":
            gamma = self.gamma_layer(time_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            beta = self.beta_layer(time_emb).unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            return x * (1 + gamma) + beta

        elif self.fusion_type == "concat":
            # Expand time embedding spatially and concat along channels
            time_emb_exp = time_emb.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
            x_cat = torch.cat([x, time_emb_exp], dim=1)  # (B, 2C, H, W)
            return self.proj_conv(x_cat)

        else:  # sum
            time_emb_exp = time_emb.unsqueeze(-1).unsqueeze(-1)  # (B, C, 1, 1)
            return x + time_emb_exp
