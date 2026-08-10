"""OneHeadDecoder configuration + the head-spec table (get_active_heads / get_active_sources).

Pure config and spec logic — no torch modules are built here. The decoder class lives in
decoder.py; _target resolves it lazily to avoid a circular import.
"""

from dataclasses import dataclass, field
from typing import List, Tuple

from queryocc.config.base_config import InstantiateConfig


def _one_head_decoder_cls() -> type:
    """Deferred import: decoder.py imports this module, so the _target default resolves lazily."""
    from queryocc.model.components.decoders.one_head.decoder import OneHeadDecoder

    return OneHeadDecoder


@dataclass
class OneHeadDecoderConfig(InstantiateConfig):
    """
    Configuration for the OneHeadDecoder.
    """

    _target: type = field(default_factory=_one_head_decoder_cls)

    freeze_decoder: bool = False
    """ Whether to freeze the decoder or not """

    hidden_dim: int = 32
    """ The dimension of the hidden features """
    offset_dim: int = 32
    """ The dimension of the offset features """
    num_resblocks: int = 3
    """ The number of resblocks """
    num_offsets: int = 1
    """ The number of offsets """
    tanh_offsets: bool = False
    """ Whether to use tanh offsets """
    tanh_offsets_scale: float = 1.0
    """ The scale of the tanh offsets """

    max_parallel_queries: int = -1  # 1_000_000
    """ The maximum number of parallel queries. Else we will shard the queries. -1 means no sharding. """
    coordinate_encoder_max_freq: float = 10.0
    """ The maximum frequency for the coordinate encoder """
    chunking_by_pre_allocate: bool = False
    """ Whether to chunk by pre-allocating the queries. """
    decoder_checkpointing: bool = False
    """ Whether to use gradient checkpointing for the decoder (only active when the model-level flag is set too). """

    occupancy: bool = True
    """ Whether to predict occupancy """

    pseudo_occupancy: bool = False
    """ Whether to predict pseudo occupancy """

    semantic_segmentation: bool = False
    """ Whether to predict semantic segmentation """
    pseudo_semantic_segmentation: bool = False
    """ Whether to predict pseudo semantic segmentation """
    supervise_free_class: bool = False
    """ Whether to supervise the free class """

    offsets_regularization: bool = False
    """ Whether to use a offsets regularization """

    # Loss weights
    occupancy_loss_weight: float = 1.0
    """ The weight of the occupancy loss """
    semantic_segmentation_loss_weight: float = 0.5
    """ The weight of the semantic segmentation loss """
    pseudo_semantic_segmentation_loss_weight: float = 0.5
    """ The weight of the pseudo semantic segmentation loss """
    offsets_regularization_loss_weight: float = 0.01
    """ The weight of the offsets regularization loss """
    pseudo_occupancy_loss_weight: float = 1.0
    """ The weight of the pseudo occupancy loss """

    @property
    def need_vfm_stuff_depth(self) -> bool:
        return self.pseudo_occupancy or self.pseudo_semantic_segmentation

    @property
    def need_vfm_stuff_semantics(self) -> bool:
        return self.pseudo_occupancy or self.pseudo_semantic_segmentation


def get_active_heads(
    config: OneHeadDecoderConfig, semantic_segmentation_classes: Tuple[str, ...]
) -> Tuple[List[str], List[int]]:
    """
    Get the active heads from the configuration.
    """
    active_heads = []
    active_dims = []
    if config.occupancy or config.pseudo_occupancy:
        active_heads.append("occupancy")
        active_dims.append(1)
    if config.semantic_segmentation or config.pseudo_semantic_segmentation:
        active_heads.append("semantic_segmentation")
        active_dims.append(len(semantic_segmentation_classes) + 2)

    return active_heads, active_dims


def get_active_sources(
    config: OneHeadDecoderConfig,
) -> List[str]:
    """
    Get the active sources from the configuration.
    """
    active_sources = []
    if config.occupancy:
        active_sources.append("occupancy")
    if config.semantic_segmentation:
        active_sources.append("semantic_segmentation")
    if config.pseudo_occupancy:
        active_sources.append("pseudo_occupancy")
    if config.pseudo_semantic_segmentation:
        active_sources.append("pseudo_semantic_segmentation")

    return active_sources
