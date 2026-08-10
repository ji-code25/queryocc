import math
from typing import List

import einops
import torch
from torch import nn
from torch.nn import functional as F

from queryocc.model.components.attention.layers import MLP, CFLayerNorm


class MSDeformAttn(nn.Module):
    """Multi-scale deformable attention op."""

    def __init__(
        self,
        dim: int,
        dim_data: int,
        num_points: int,
        num_heads: int,
        num_scales: int,
        init_head_directional: bool = False,
        align_corners: bool = False,
        prenorm_spatial_data: bool = False,
    ):
        """Initialize.

        Args:
            dim: Dimensionality of latent and output.
            dim_data: Dimensionality of the data representation to cross-attend. In the original
                      paper this is always equal to dim.
            num_points: Number of points to sample. Can be seen as the number of 'keys' a given
                        query will attend to.
            num_heads: Number of attention heads.
            num_scales: Number of resolution scales to attend to.
            init_head_directional: Whether to initialize the sampling offsets to point in different
                                   directions for different heads (True) or different points
                                   (False). If True (False) the points (heads) will be initialized
                                   for progressively further sampling. True corresponds to the
                                   original paper and False is custom designed in-house. The
                                   in-house initialization approach seems to yield superior
                                   results.
            align_corners: Set to True when using projection matrices to get pixel coordinates for
                           2D points. Set to False when using 3D or BEV grid coordinates.
            prenorm_spatial_data: Whether to fuse a prenormalization of the spatial data into this
                                  operation. This is more efficient than running it outside becaue
                                  it can be absorbed in the inner for loop of this layer.
        """
        assert dim % num_heads == 0, (
            f"The model latent dimensionality (={dim}) has to be divisible by the number of attention heads (={num_heads})."
        )
        super().__init__()

        # Store for convenience.
        self._num_heads = num_heads
        self._num_points = num_points
        self._num_scales = num_scales

        # Value projection function is defined as a 1x1 convolution rather than a Linear layer.
        # Mathematically they behave equivalently, but a convolution allows us to operate on a
        # channels-first tensor, which alligns with that F.grid_sample expects for interpolating
        # feature maps. This is absolutely performance-critical for the layer, and using a Linear
        # here results in at least a >2x slowdown of the entire layer.
        if prenorm_spatial_data:
            # Note, do not try to fuse this with torch.compile. I tested it and it becomes 1.7x
            # slower.
            self._proj_v = nn.Sequential(CFLayerNorm(dim_data), nn.Conv2d(dim_data, dim, kernel_size=1))
        else:
            self._proj_v = nn.Conv2d(dim_data, dim, kernel_size=1)

        # Output projection for mixing the attention heads.
        self._proj_o = nn.Linear(dim, dim)

        # 2D offsets for sampling each point per head and scale + one attention score for each
        # sampled point at each scale, with separate attention scores between the heads.
        # We predict both in one fused operation for efficiency.
        self._fused_proj = nn.Linear(dim, num_heads * num_scales * num_points * 3)
        self._split_index = num_heads * num_scales * num_points * 2

        self._align_corners = align_corners

        # Initialize all weights.
        self.reset_parameters(init_head_directional=init_head_directional)

    def _initialize_fused_proj_weights(self, init_head_directional: bool) -> None:
        # We want the offsets from the sampling locations to starts as constant values, with
        # sampling happening in each direction. Each attention head samples in its own direction.
        # Alternatively each head samples concentrically further away from the reference point. In
        # either case we want the offsets to start independently from the model activations so we
        # set the weights to zero. For attention scores we also want to start with a uniform
        # attention prior so we also want zero weights.
        nn.init.constant_(self._fused_proj.weight.data, 0.0)

        # Compute concentric angles along the unit circle.
        num = self._num_heads if init_head_directional else self._num_points
        angles = (2 * math.pi) * torch.arange(num, dtype=next(self._fused_proj.parameters()).dtype) / num

        # Note that grid_sample expects [w, h] instead of [h, w] points.
        init_sample_loc = torch.stack([angles.cos(), angles.sin()], dim=-1)

        # Instead of sampling on a circle we want to sample on a square because the centers of
        # diagonal grid cells are further away than those od adjecent ones.
        init_sample_loc /= init_sample_loc.abs().max(-1, keepdim=True)[0]

        # Sample progressively further from the reference for each of the points.
        # All scales have the same initialization values, but because they represent different
        # resolution scales, they actually correspond to different scales in the real world.
        # (AH, 2) -> (AH, S, P, 2)
        if init_head_directional:
            init_sample_loc = init_sample_loc.view(self._num_heads, 1, 1, 2) * torch.arange(
                1,
                self._num_points + 1,
                # TODO The last repeat only has to happen once for both clauses.
            ).view(1, 1, self._num_points, 1).repeat(1, self._num_scales, 1, 1)
        else:
            init_sample_loc = init_sample_loc.view(1, 1, self._num_points, 2) * torch.arange(
                1, self._num_heads + 1
            ).view(self._num_heads, 1, 1, 1).repeat(1, self._num_scales, 1, 1)
        self._fused_proj.bias.data[: self._split_index] = init_sample_loc.view(-1)

        # We want the attention to start uniform over all sampled points. We can get this by
        # initializing biases to zeroes. The weights were initialized to zeroes above.
        self._fused_proj.bias.data[self._split_index :] = 0

    def reset_parameters(self, init_head_directional: bool = True) -> None:
        """Initialize all model parameters."""
        self._initialize_fused_proj_weights(init_head_directional=init_head_directional)

        # Because transformers do not rely on ReLU it is better to initialize the projections for
        # attention with Xavier/Glorot than Kaiming. Note that this is not the case for the FFN.
        # Pytorch defaults to Kaiming.
        pv = self._proj_v[1] if isinstance(self._proj_v, nn.Sequential) else self._proj_v
        nn.init.xavier_uniform_(pv.weight.data)
        nn.init.xavier_uniform_(self._proj_o.weight.data)

    def _prepare_ref_points_scale(self, ref_points: torch.Tensor) -> torch.Tensor:
        """Prepare the reference points for summing with predicted offsets."""
        # Deformable attention from Deformable DETR.
        if len(ref_points.shape) == 3:
            # (B, N, 2) -> (B*AH, N, P, 2)
            # TODO: Is there a way to use 'expand' here instead to avoid a copy of this tensor? It
            # Doesn't seem to work on non-singleton tensors.
            return ref_points.unsqueeze(2).repeat(self._num_heads, 1, self._num_points, 1)

        # Pointwise deformable attention from StreamMapNet. Each offset has its own reference
        # point (though they are equal for the different attention heads).
        if len(ref_points.shape) == 4:
            # (B, N, P, 2) -> (B*AH, N, P, 2)
            return ref_points.repeat(self._num_heads, 1, 1, 1)

        raise NotImplementedError(
            f"Reference points must be either a tensor of rank 3 (deformable DETR) or rank 4 (StreamMapNet) but got rank {len(ref_points.shape)}."
        )

    def forward(
        self, latent: torch.Tensor, ref_points: torch.Tensor, spatial_data_scales: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            latent: (B, N, C).
            ref_points: (B, N, S, 2) in for normal deformable attention or (B, N, S, P, 2) for
                        point-wise deformably attehtion. Expected to be normalized to the range
                        [-1, 1]. By enforcing normalization this will work both with unit pixel
                        ranges and voxel grids.
            spatial_data_scales: List of (B, D, H, W) tensors that will be attended. It is
                                 performance-critical that the spatial tensors are in a
                                 channels-first configuration.
        Returns:
            Update to latent, of shape (B, N, C)
        """
        # We use the following notation convention in the implementation:
        # b: batch size
        # n: Number of latents/queries
        # s: Number of resolution scales.
        # p: Number of points.
        # ah: attention heads
        # hc: head channels
        # h: image height (spatial data x)
        # w: image width (spatial data y)
        # t: 2 (einops does not accept literals other than 1)

        assert latent.shape[1] == ref_points.shape[1], (
            f"Expected each latent to have a reference point but got {latent.shape[1]} latents and {ref_points.shape[1]} reference points."
        )
        assert len(spatial_data_scales) == ref_points.shape[2], (
            f"Expected a reference point for each data scale, but got {len(spatial_data_scales)} data scales and {ref_points.shape[2]} sets of reference points."
        )
        assert ref_points.shape[2] == self._num_scales, (
            f"Expected the initialized number of scales (={self._num_scales}) to be equivalent to the number of scales passed to this layer (={ref_points.shape[2]})."
        )
        assert len(ref_points.shape) in (
            4,
            5,
        ), (
            f"Expected the reference point tensor to have rank 4 (standard deformable attention) or rank 5 (pointwise deformable attention) but got rank {len(ref_points.shape)}"
        )
        assert len(ref_points.shape) == 4 or ref_points.shape[3] == self._num_points, (
            f"Expected the number of reference points in pointwise deformable attention to be equal to the number of predicted offsets but got {ref_points.shape[3]} and {self._num_points} respectively."
        )

        # Fused prediction of attention scores and offsets in one go.
        fused_offset_attn = self._fused_proj(latent)

        # Isolate the heads along the batch axis for later bilinear sampling.
        # Note that it is more efficient to have S first so that the indexed tensor in the loop
        # below is contiguous in memory!
        # (B, N, AH*S*P*2) -> (S, B*AH, N, P, 2).
        sample_points = einops.rearrange(
            fused_offset_attn[..., : self._split_index],
            "b n (ah s p t) -> s (b ah) n p t",
            ah=self._num_heads,
            s=self._num_scales,
            t=2,
        )

        B, N, _, _ = ref_points.shape if len(ref_points.shape) == 4 else ref_points.shape[:-1]
        C = self._fused_proj.in_features

        # Mix the attention heads with the projection.
        return self._proj_o(
            (
                # Features will first be aggregated into a (S, B*AH, HC, N, P) tensor for
                # efficiency (see the stack below) and then reshaped as follows:
                # (S, B*AH, HC, N, P) -> (B, N, S*P, AH, HC)
                # so that we can do an attention-weighted sum per head over the points and scales
                # before concatenating the heads. Note that (S*P, AH, HC) is significantly more
                # efficient than (AH, S*P, HC) since the former can implicitly concatenated as a
                # memory view.
                einops.rearrange(
                    # Given that the sampling point tensor starts with the S dimension we can stack
                    # all the scales together and then reshape it back in the attention sum below.
                    torch.stack(
                        [
                            # Each scale will sample at (B*AH, HC, N, P) and later we use attn
                            F.grid_sample(
                                # View is slightly faster than einops here since it avoids some
                                # overhead. Note that the projection here operates on a
                                # channels-first tensor because it is defined as a convolution.
                                # This is highly performance critical since the view below can not
                                # be done in channels-last alignment to make a valid grid-sample
                                # input. This has been verified to result in a >2x slowdown of the
                                # entire layer.
                                input=self._proj_v(sds).reshape(  # NOT: Adli changed this from view to reshape
                                    B * self._num_heads,
                                    C // self._num_heads,
                                    sds.shape[2],
                                    sds.shape[3],
                                ),
                                # Warning: Do not hardcode fp32 for AMP training but use the dtype
                                # of the inputs.
                                # Warning: The division by the predicted sample points by the
                                # tensor scales is crucial for stable layer performance.
                                grid=(
                                    self._prepare_ref_points_scale(ref_points[:, :, i, :])
                                    # Multiply by a factor 2 because the range is [-1, 1]. If we
                                    # predict an offset of H we want to shift the sampling
                                    # coordinate by 2 points.
                                    + 2
                                    * sample_points[i]
                                    / torch.tensor(
                                        # Remember that 2D coordinates passed to grid sample are
                                        # (w, h) for a (H, W) tensor.
                                        (sds.shape[3], sds.shape[2]),
                                        dtype=sample_points.dtype,
                                        device=sample_points.device,
                                    ).view(1, 1, 1, 2)
                                ),
                                mode="bilinear",
                                # Zero-padding results in faster predictions out of bounds.
                                padding_mode="zeros",
                                align_corners=self._align_corners,
                            )
                            for i, sds in enumerate(spatial_data_scales)
                        ],
                        dim=0,
                    ),
                    # Reshape S and AH back so we can do a head-wise attention summation over all
                    # scales and points.
                    "s (b ah) hc n p -> b n (s p) ah hc",
                    ah=self._num_heads,
                )
                # Extract attention scores from the fused offset + attention predictions and weight
                # each point across the scales before summing them. Note that explicit
                # multiplication followed by sum is infinitesimally faster than "torch.einsum" here
                # because the operation gets fused and avoids einsum overhead.
                * F.softmax(
                    fused_offset_attn[..., self._split_index :].view(
                        B, N, self._num_scales * self._num_points, self._num_heads
                    ),
                    dim=2,
                ).unsqueeze(-1)
            )
            .sum(dim=2)
            .view(B, N, C)
        )


class MSDeformCrossAttnBlock(nn.Module):
    """Single stage Deformable cross attention block.

    As opposed to the implementation of the deformable DETR paper, this implementation is
    prenormalized, which we found to work much better in practice.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        dim: int,
        dim_data: int,
        num_points: int,
        num_heads: int,
        num_scales: int,
        init_head_directional: bool = False,
        align_corners: bool = False,
        ff_mult: int = 4,
        activation=nn.ReLU(inplace=True),
        prenorm_spatial_data: bool = False,
    ):
        """Initialize.

        Args:
            See DeformAttn for most arguments.
            ff_mult: Feedforward multiplier.
            activation: Activation function.
            prenorm_spatial_data: Whether to prenormalize the spatial data. If you want to use it
                                  we recommend only using it for the first layer in a number of
                                  transformer blocks. LayerNorm is relatively expensive.
        """
        super().__init__()
        # TODO We need a custom channels-first layernorm operator if we also want to prenorm the
        # spatial tensors.
        self._q_norm = nn.LayerNorm(dim)
        self._ffn_norm = nn.LayerNorm(dim)
        self._deform_attn = MSDeformAttn(
            dim=dim,
            dim_data=dim_data,
            num_points=num_points,
            num_heads=num_heads,
            num_scales=num_scales,
            init_head_directional=init_head_directional,
            align_corners=align_corners,
            prenorm_spatial_data=prenorm_spatial_data,
        )
        self._ffn = MLP(
            input_dim=dim,
            hidden_dim=ff_mult * dim,
            output_dim=dim,
            num_layers=2,
            activation=activation,
        )

    def forward(
        self,
        latent: torch.Tensor,
        ref_points: torch.Tensor,
        spatial_data_scales: List[torch.Tensor],
    ) -> torch.Tensor:
        """Apply deformable cross attention to update the latent."""
        # Note that the spatial_data_scales are prenormalized in a fused way when requested in the
        # initialization of this layer. Applying LayerNorm here would be slower.
        latent = latent + self._deform_attn(self._q_norm(latent), ref_points, spatial_data_scales)

        # TODO Cross-check what kind of FFN, if any, they use in the Deformable DETR paper and in
        # BEVFormer.
        return latent + self._ffn(self._ffn_norm(latent))


def _grid_ref_points(spatial_data: torch.Tensor, num_scales: int = 1) -> torch.Tensor:
    """Build the reference points for self-attending spatial tensors."""
    # Note that even though we pass coordinates as (w, h) for a (H, W) tensor we need to do the
    # meshgrid in H, W order and flip the axes later. Otherwise points will get the wrong reference
    # when tensor indices step in height for dimension 0 and in width for the meshgrids 0
    # dimension.
    # We shift by 0.5 to get the correct grid center positions.
    B, _, H, W = spatial_data.shape
    ref = torch.stack(
        torch.meshgrid(
            (torch.arange(H, dtype=torch.float32) + 0.5) / H,
            (torch.arange(W, dtype=torch.float32) + 0.5) / W,
            indexing="ij",
        ),
        dim=-1,
    ).to(device=spatial_data.device)
    # Remember that grid_sample in torch is fucked and we need to pass width coordinates first
    # even though that obviously doesn't make any sense. We have to flip the last indices.
    ref = ref.flip(-1)

    # (H, W, 2) -> (1, H*W, 2) -> (B, H*W, 2)
    # Use expand here instead of view not to assign a new tensor.
    return (ref * 2 - 1).view(1, H * W, 1, 2).expand(B, H * W, num_scales, 2)


class MSDeformSelfAttnBlockOneToAll(nn.Module):
    """Deformable self-attention block.

    One data scale attends the rest of the given data scales. Reference points are built
    hierarchically for every scale in an implicit way.

    Prenormed like the cross attention block.
    """

    def __init__(self, dim: int, **kwargs):
        """Initialize.

        Args:
            See MSDeformCrossAttnBlock.
        """
        super().__init__()

        # Hardcore corner alignment since the meshgrid we compute in the forward assumes the
        # corners are not aligned. If we would not do align_corners=False here the normalized
        # coordinates would become inconsistent between data scales.
        self._attn_block = MSDeformCrossAttnBlock(dim=dim, dim_data=dim, align_corners=False, **kwargs)

    def forward(self, spatial_data_scales: List[torch.Tensor], self_index: int) -> List[torch.Tensor]:
        """Apply deformable self-attention on a 2D feature grid attending all other scales.

        Args:
            spatial_data_scales: List of of (B, C, H, W) spatial tensors.
            self_index: The index of the spatial map that will be the latent that is updated based
                        on attending all scales (including itself).

        Returns:
            List of [(B, C, H, W) tensors where only index=self_index is updated.
        """
        spatial_latent = spatial_data_scales[self_index]
        # TODO We can precompute these.
        ref_points = _grid_ref_points(spatial_latent, num_scales=len(spatial_data_scales))

        # Reshape the spatial latent.
        # TODO: Make a formulation of MSDeformAttn that operates channels-first for extra
        # efficiency. This can be done with nn.Conv1D.
        _, _, H, W = spatial_latent.shape
        spatial_latent = einops.rearrange(spatial_latent, "b c h w -> b (h w) c")
        spatial_latent = self._attn_block(
            latent=spatial_latent, ref_points=ref_points, spatial_data_scales=spatial_data_scales
        )

        # Put back the spatial latent into the input and return so that this layer can be easily
        # stacked.
        # TODO: This can be done with a memory view if we can operate on a channels-first latent.
        spatial_data_scales[self_index] = einops.rearrange(spatial_latent, "b (h w) c -> b c h w", h=H, w=W)
        return spatial_data_scales


class MSDeformSelfAttnBlockAllToAll(MSDeformSelfAttnBlockOneToAll):
    """Multi-scale deformable self-attention between all spatial scales."""

    # pylint: disable=arguments-differ
    def forward(self, spatial_data_scales: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward.

        Args:
            spatial_data_scales: List of of (B, C, H, W) spatial tensors that will all attend to
                                 each other with MS deformable attention.

        Returns:
            List of [(B, C, H, W) tensors that all attended each other.
        """
        # TODO All the split indices etc could be precomputed.
        hws = [(sds.shape[2], sds.shape[3]) for sds in spatial_data_scales]
        sizes = [h * w for h, w in hws]
        split_indices = [0] + [sum(sizes[: i + 1]) for i in range(len(sizes))]
        split_indices = list(zip(split_indices[:-1], split_indices[1:]))
        spatial_latent = torch.cat(
            [einops.rearrange(sds, "b c h w -> b (h w) c") for sds in spatial_data_scales], dim=1
        )
        ref_points = torch.cat(
            [_grid_ref_points(sds, num_scales=len(spatial_data_scales)) for sds in spatial_data_scales],
            dim=1,
        )
        spatial_latent = self._attn_block(
            latent=spatial_latent, ref_points=ref_points, spatial_data_scales=spatial_data_scales
        )
        return [
            einops.rearrange(spatial_latent[:, start:end, :], "b (h w) c -> b c h w", w=w, h=h)
            for (start, end), (h, w) in zip(split_indices, hws)
        ]
