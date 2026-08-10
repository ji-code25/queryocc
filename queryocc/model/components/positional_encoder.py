import math
from typing import List

import torch
from torch import nn
from torch.profiler import record_function


def _frequencies_orig(embed_dim: int, pos_dim: int, max_freq: float) -> torch.Tensor:
    """Frequencies used in the positional encoding of the original transformer paper.
    See Attention Is All You Need (https://arxiv.org/abs/1706.03762). These positional encodings
    are also used in DETR (https://arxiv.org/abs/2005.12872).
    Args:
        embed_dim: Dimensionality of the embeddings. The positional encoding will have the same
                   dimensionality.
        pos_dim: Dimensionality of the positions to encode. For instance 1 for text, 2 for images
                 and 3 for lidar.
        max_freq: Maximum frequency value. 1e4 in the original paper.
    Returns:
        Frequencies for positional encoding. Shape (embed_dim,)
    """
    log = math.log(max_freq)
    axis_dim = math.ceil(embed_dim / pos_dim)
    fqs = torch.arange(axis_dim // 2, dtype=torch.float)
    fqs = torch.exp(-2 * (fqs / axis_dim) * log)
    return fqs


def _frequencies_perceiver(embed_dim: int, max_freq: float) -> torch.Tensor:
    """Frequencies used in the positional encoding of the Perceiver paper.
    See https://arxiv.org/abs/2103.03206. In Perceiver, the input positions are expected to be
    normalized to the [-1, 1] interval. Account for this when using this.
    Args:
        embed_dim, max_freq: See _frequencies_orig.
    Returns:
        See _frequencies_orig
    """
    # In the original perceiver paper the number of frequencies is a hyperparamter
    fqs = torch.pi * torch.linspace(0, max_freq, embed_dim // 2, dtype=torch.float).flip(dims=[-1])
    return fqs


def _frequencies(embed_dim: int, pos_dim: int, max_freq: float, perceiver: bool) -> torch.Tensor:
    """Route to the correct frequency function."""
    if perceiver:
        return _frequencies_perceiver(embed_dim, max_freq)
    return _frequencies_orig(embed_dim, pos_dim, max_freq)


def build_static_grid_encoding(
    embed_dim: int, axis_sizes: List[int], max_freq: float, perceiver: bool = False
) -> torch.Tensor:
    """Precomputed interleaved sin/cos encoding for a fixed grid, (prod(axis_sizes), embed_dim).

    The single implementation behind the BEV-canvas initializations (lidar voxelizer, LSS lifter,
    VGGT aggregator) — this block used to be hand-copied at each site with only max_freq differing.
    """
    dim_offset = embed_dim % (2 * len(axis_sizes))
    fqs = _frequencies(embed_dim + dim_offset, len(axis_sizes), max_freq, perceiver).unsqueeze(
        0
    )  # (1, embed_dim/2*pos_dim)
    # Make positional encodings along each dimension.
    axis_pes = []
    for ax in axis_sizes:
        pos = torch.arange(ax, dtype=torch.float).unsqueeze(1)  # (pos, 1)
        pe = pos * fqs
        axis_pes.append(torch.cat([torch.sin(pe), torch.cos(pe)], dim=-1))
    # Expand all the 1D positional encodings so they can be stacked along all dimensions.
    for i, _ in enumerate(axis_pes):
        repeats = list(axis_sizes).copy() + [1]
        repeats[i] = 1
        view_shape = [1 for _ in axis_sizes] + [axis_pes[i].shape[-1]]
        view_shape[i] = axis_sizes[i]
        axis_pes[i] = axis_pes[i].view(*view_shape).repeat(*repeats)
    # Stack and flatten instead of concatenate to interleave the dimensions. This allows a
    # clean dimensionality cutoff that does not favor any particular dimension.
    pe_full = torch.stack(axis_pes, dim=-1)  # (X1, X2, .., XN, embed_dim/N, N)
    pe_full = pe_full.flatten(start_dim=-2, end_dim=-1)
    # Now make a cutoff if needed.
    return pe_full[..., :embed_dim].reshape(-1, embed_dim)


class CoordinateEncoder(nn.Module):
    """Fourier features + MLP over arbitrary coordinates (time, points, calibration descriptors).

    Moved here from decoders/uno_decoder.py: it is a generic positional encoder used by encoders
    and decoders alike (the old location forced encoder->decoder imports). uno_decoder re-exports
    it for backward compatibility.
    """

    def __init__(
        self,
        coordinate_dim: int,
        hidden_dim: int,
        out_dim: int,
        num_freqs: int,
        min_freq: float = 1.0,
        max_freq: float = 8.0,
    ):
        super(CoordinateEncoder, self).__init__()

        self._coordinate_dim = coordinate_dim
        self._hidden_dim = hidden_dim
        self._out_dim = out_dim
        self._num_freqs = num_freqs

        _freqs_log = torch.linspace(math.log(min_freq), math.log(max_freq), num_freqs // 2)
        _freqs = torch.exp(_freqs_log)

        self.register_buffer(
            "_freqs",
            _freqs,
        )

        self.mlp = nn.Sequential(
            nn.Linear(coordinate_dim * num_freqs, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )

    @record_function("CoordinateEncoder.forward")
    def forward(self, coordinates: torch.Tensor) -> torch.Tensor:
        encoded = self.sinusoidal_embedding(coordinates)  # B, N, C
        B, N, _ = encoded.shape
        # Reshape, apply MLP, then restore dimensions
        encoded = encoded.view(-1, encoded.size(-1))  # (B*N), C
        encoded = self.mlp(encoded)  # (B*N), out_dim
        return encoded.view(B, N, -1)  # B, N, out_dim

    def sinusoidal_embedding(self, coordinates: torch.Tensor) -> torch.Tensor:
        # coordinates: B, N, C
        # B, N, C
        B, N, _ = coordinates.size()
        # B, N, C, sinusoide_dim // 2
        sin = torch.einsum("bnc,d->bncd", coordinates, self._freqs)
        # B, N, C, sinusoide_dim
        sin = torch.cat([sin.sin(), sin.cos()], dim=-1)
        # B, N, C * sinusoide_dim
        return sin.view(B, N, -1)
