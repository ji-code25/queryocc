from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Union

import torch
from torch import Tensor

from queryocc.common.types import PointCloudRange, VoxelSize
from queryocc.config.base_config import InstantiateConfig
from queryocc.config.logging_configs import LoggingConfig
from queryocc.data.augmentations.augmentation_configs import AugmentationConfig
from queryocc.data.datasets.nuscenes_mappings import DEFAULT_NUSCENES_CLASSES, LabelMapper
from queryocc.model.components.contraction.point_contraction import PointContractionConfig
from queryocc.model.components.decoders.one_head import OneHeadDecoderConfig
from queryocc.model.components.image_encoders.cerveza.encoder import CervezaImageEncoderConfig
from queryocc.model.components.uno_encoder import UnoBEVEncoder


@dataclass
class NetConfig(InstantiateConfig):
    _target: type = field(default_factory=lambda: QueryOccNet)
    """The target class for instantiation."""

    compile: bool = False
    """Whether to compile the model with torch.compile()."""
    gradient_checkpointing: bool = False
    """Whether to use gradient checkpointing."""
    swap_batch_norm_to_instance_norm: bool = False
    """Whether to swap batch norm to instance norm."""

    hidden_dim: int = 128

    # Encoders
    lidar_encoder: bool = False
    """ Lidar input encoder — not supported by QueryOcc (camera-only model; lidar is a
    supervision source, not an input). Must stay False. """
    image_encoder: bool = True
    """ Whether to use the image encoder (required). """

    image_encoder_name: Literal["cerveza"] = "cerveza"
    """ The name of the image encoder to use """

    cerveza_encoder: CervezaImageEncoderConfig = field(default_factory=CervezaImageEncoderConfig)
    """ The configuration for the Cerveza image encoder """

    lifter_canvas_size: Tuple[int, int, int] = field(init=False)  # will be set by the parent model
    """ The canvas size for the lifter """
    bev_features_canvas_size: Tuple[int, int] = field(init=False)  # will be set by the parent model
    """ The canvas size for the encoded BEV features """

    # BEV Processing
    bev_attention: Literal["none", "deformable_py"] = "deformable_py"
    """ Whether to use deformable attention in the encoder model """
    resblock_convs: Literal["dynamic", "standard"] = "dynamic"
    """ Whether to use dynamic or standard resblock convolutions in the encoder model """
    attention_layers: int = 6
    """ The number of attention layers in the encoder model """

    # Semantic Segmentation
    semantic_segmentation_classes: Tuple[str, ...] = DEFAULT_NUSCENES_CLASSES
    """ The classes to use for semantic segmentation """
    semantic_segmentation_classwise_weights: Literal["none", "class_frequency", "class_frequency_pseudo"] = "none"
    """ Whether to use classwise weights for the semantic segmentation loss """
    semantic_segmentation_classwise_weights_free_frequency_factor: float = 1.0
    """ The factor to multiply the free frequency by. A smaller number will give more weight to the free class. 1.0 means equal weight with the combination of all other classes."""
    semantic_segmentation_classwise_weights_alpha: float = 1.0
    """ The alpha parameter for the classwise weights. For alpha > 1, rare classes get disproportionately higher weights. For alpha < 1, the imbalance effect is dampened"""

    # One head
    use_one_head: bool = True
    """ Whether to use the unified query decoder (required). """
    one_head: OneHeadDecoderConfig = field(default_factory=OneHeadDecoderConfig)
    """ Configuration for the one head decoder """

    use_point_contraction: bool = False
    """ Whether to use point contraction """
    point_contraction: PointContractionConfig = field(default_factory=PointContractionConfig)
    """ Configuration for the point contraction """

    @property
    def f_use_occupancy_sampler(self) -> bool:
        return self.use_one_head

    @property
    def need_vfm_stuff(self) -> bool:
        return self.use_one_head and (self.one_head.pseudo_occupancy or self.one_head.pseudo_semantic_segmentation)

    @property
    def init_point_contraction(self) -> bool:
        return self.cerveza_encoder.lss.use_point_contraction or self.use_point_contraction

    @property
    def need_offset_pointclouds(self) -> bool:
        return self.use_one_head and self.one_head.occupancy


def sanity_check_net_config(config: NetConfig):
    assert not config.lidar_encoder, "QueryOcc is camera-only: the lidar input encoder is not supported."
    assert config.image_encoder, "QueryOcc requires the image encoder."
    assert config.use_one_head, "QueryOcc requires the unified one-head query decoder."


class QueryOccNet(torch.nn.Module):
    def __init__(
        self,
        config: NetConfig,
        point_cloud_range: PointCloudRange,
        voxel_size: VoxelSize,
        logging_config: LoggingConfig,
        augmentation_config: AugmentationConfig,
        min_query_time: float,
        max_query_time: float,
    ):
        super().__init__()
        sanity_check_net_config(config)
        self.config = config
        self.logging_config = logging_config
        self.augmentation_config = augmentation_config
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size

        if config.init_point_contraction:
            self.point_contraction = config.point_contraction.setup(point_cloud_range=point_cloud_range)
        else:
            self.point_contraction = None

        self.image_encoder = config.cerveza_encoder.setup(
            feature_dim=config.hidden_dim,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            lifter_canvas_size=self.config.lifter_canvas_size,
            point_contraction=self.point_contraction,
            compile=config.compile,
            gradient_checkpointing=config.gradient_checkpointing,
            swap_batch_norm_to_instance_norm=config.swap_batch_norm_to_instance_norm,
            logging_config=logging_config,
            augmentation_config=augmentation_config,
        )

        self.bev_encoder = UnoBEVEncoder(
            feature_dim=config.hidden_dim,
            attention=config.bev_attention,
            resblock_convs=config.resblock_convs,
            attention_layers=config.attention_layers,
            gradient_checkpointing=config.gradient_checkpointing,
        )
        if config.compile:
            # In-place compile keeps state_dict keys free of the `_orig_mod.` prefix,
            # so checkpoints stay interchangeable between compiled and eager runs.
            # dynamic=False on all compile sites: input shapes are fixed by the ROI/voxel
            # grid and sampler configs, and symbolic shapes break Inductor on the
            # multi-scale deformable attention (CantSplit on sum-of-products sizes) when
            # the DDP graph splitter recompiles under training. Distinct shapes (e.g.
            # eval vs train) just compile separate static graphs.
            self.bev_encoder.compile(dynamic=False)

        semseg_weights = get_semseg_weights(config)
        self.one_head_decoder = config.one_head.setup(
            input_dim=config.hidden_dim,
            point_cloud_range=point_cloud_range,
            gradient_checkpointing=config.gradient_checkpointing,
            semantic_segmentation_classes=config.semantic_segmentation_classes,
            semantic_segmentation_weights=semseg_weights,
            use_point_contraction=config.use_point_contraction,
            point_contraction=self.point_contraction,
            min_query_time=min_query_time,
            max_query_time=max_query_time,
        )
        if config.compile:
            # The inner decoder is where the per-query work happens (hundreds of
            # thousands of queries per step); the OneHeadDecoder wrapper around it does
            # dict/slice bookkeeping that is not worth tracing. Note: for UnoDecoder only
            # the unchunked path (max_parallel_queries=-1) goes through the compiled
            # __call__ (forward_chunks calls .forward() directly and stays eager).
            self.one_head_decoder._decoder.compile(dynamic=False)

    @property
    def can_decode_semantic_segmentation(self) -> bool:
        return self.config.use_one_head and (
            self.config.one_head.semantic_segmentation or self.config.one_head.pseudo_semantic_segmentation
        )

    @property
    def can_decode_occupancy(self) -> bool:
        return self.config.use_one_head and (self.config.one_head.occupancy or self.config.one_head.pseudo_occupancy)

    def get_encoders(self) -> List[torch.nn.Module]:
        return [e for e in [self.image_encoder, self.bev_encoder] if e is not None]

    def forward(self, lidar: dict, camera: dict) -> Tuple[Tensor, dict]:
        return self.encode(lidar, camera)

    def encode(self, lidar: dict, camera: dict) -> Tuple[Tensor, dict]:
        encoding, info = self.image_encoder(**camera)
        encoding, info = self._bev_encode(encoding, info)
        return encoding, info

    def _bev_encode(self, encoding: Tensor, info: dict) -> Tuple[Tensor, dict]:
        return self.bev_encoder(encoding), info

    def decode(
        self, encoding: Tensor, queries: Tensor, to_decode: list[str], max_queries: Optional[int] = None, **kwargs
    ) -> Dict[str, torch.Tensor]:
        out = {}

        one_out = self.one_head_decoder.decode(encoding, queries, max_parallel_queries=max_queries)
        if "occupancy" in to_decode:
            out["occupancy"] = one_out["occupancy_logits"]
        if "semantic_segmentation" in to_decode:
            if "semantic_segmentation_logits" not in one_out:
                raise ValueError(f"Semantic segmentation logits not found in one_head_decoder for {to_decode}")
            semseg_logits = one_out["semantic_segmentation_logits"]

            occupancy_logits = one_out["occupancy_logits"].detach()
            del one_out

            # Calculate the probability of being occupied (P_occupied).
            occupancy_probs = torch.sigmoid(occupancy_logits)
            del occupancy_logits

            # Separate the logits for the *non-free* classes.
            non_free_logits = semseg_logits[..., :-1]

            # Calculate the conditional probabilities P(Class_i | Occupied).
            # These are the softmax probabilities over the non-free classes.
            non_free_probs = torch.softmax(non_free_logits, dim=-1)
            del non_free_logits

            # Calculate the joint probability for non-free classes:
            # P(Class_i, Occupied) = P(Class_i | Occupied) * P(Occupied)
            # This is equivalent to P(Class_i) since Class_i implies Occupied.
            non_free_probs *= occupancy_probs

            # Calculate the probability of the 'free' class (P_free).
            free_probs = 1 - occupancy_probs

            # Concatenate the probabilities to form the final set.
            semseg_probs = torch.cat([non_free_probs, free_probs], dim=-1)

            out["semantic_segmentation_logits"] = semseg_logits
            out["semantic_segmentation_probs"] = semseg_probs
            out["semantic_segmentation_preds"] = semseg_probs.argmax(dim=-1, keepdim=True)

        return out


def get_semseg_weights(config: NetConfig) -> Union[Tensor, None]:
    label_mapper = LabelMapper(config.semantic_segmentation_classes)
    if config.semantic_segmentation_classwise_weights == "class_frequency":
        class_frequencies = label_mapper.learning_to_num_points(("noise",) + config.semantic_segmentation_classes)
        free_frequency = sum(class_frequencies) * config.semantic_segmentation_classwise_weights_free_frequency_factor
        frequencies = torch.tensor(class_frequencies + [free_frequency])
        log_weights = torch.log(frequencies.sum() / frequencies)
        semseg_weights = log_weights**config.semantic_segmentation_classwise_weights_alpha
        semseg_weights = (log_weights.numel() / semseg_weights.sum()) * semseg_weights

    elif config.semantic_segmentation_classwise_weights == "class_frequency_pseudo":
        class_frequencies = label_mapper.learning_to_num_points_pseudo(
            ("noise",) + config.semantic_segmentation_classes
        )
        free_frequency = sum(class_frequencies) * config.semantic_segmentation_classwise_weights_free_frequency_factor
        frequencies = torch.tensor(class_frequencies + [free_frequency])
        log_weights = torch.log(frequencies.sum() / frequencies)
        semseg_weights = log_weights**config.semantic_segmentation_classwise_weights_alpha
        semseg_weights = (log_weights.numel() / semseg_weights.sum()) * semseg_weights

    else:
        semseg_weights = None

    return semseg_weights
