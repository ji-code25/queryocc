"""OneHeadDecoder: the primary multi-task decode head (see one_head/__init__.py).

This module holds the class skeleton: construction, head bookkeeping, query assembly,
decode, and the forward loss dispatch. Loss families live in the method mixin
(losses.py) — mixins hold methods only, never parameters.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from queryocc.common.types import PointCloudRange
from queryocc.data.datasets.nuscenes_mappings import DEFAULT_NUSCENES_CLASSES, LabelMapper
from queryocc.model.components.contraction.point_contraction import PointContraction
from queryocc.model.components.decoders.one_head.config import (
    OneHeadDecoderConfig,
    get_active_heads,
    get_active_sources,
)
from queryocc.model.components.decoders.one_head.losses import QueryLossesMixin
from queryocc.model.components.decoders.uno_decoder import UnoDecoder


class OneHeadDecoder(QueryLossesMixin, nn.Module):
    def __init__(
        self,
        config: OneHeadDecoderConfig,
        input_dim: int,
        point_cloud_range: PointCloudRange,
        gradient_checkpointing: bool = False,
        semantic_segmentation_classes: Tuple[str, ...] = DEFAULT_NUSCENES_CLASSES,
        semantic_segmentation_weights: Optional[torch.Tensor] = None,
        use_point_contraction: bool = False,
        point_contraction: Optional[PointContraction] = None,
        min_query_time: float = -3.0,
        max_query_time: float = 3.0,
        pc_dim: int = 4,
    ):
        super().__init__()
        self.config = config
        self.semantic_segmentation_classes = semantic_segmentation_classes

        ah, ad = get_active_heads(config, semantic_segmentation_classes)
        self.active_heads = ah
        self.active_dims = ad
        total_dims = sum(ad)

        # Each head owns a contiguous channel range of the shared prediction tensor;
        # slices let us read them as views instead of index_select copies.
        head_to_dims = {}
        start_dim = 0
        for head, dim in zip(ah, ad):
            end_dim = start_dim + dim
            head_to_dims[head] = slice(start_dim, end_dim)
            start_dim = end_dim
        self.head_to_dims = head_to_dims

        self._decoder = self._build_decoder(
            input_dim=input_dim,
            total_dims=total_dims,
            point_cloud_range=point_cloud_range,
            gradient_checkpointing=gradient_checkpointing,
            use_point_contraction=use_point_contraction,
            point_contraction=point_contraction,
            pc_dim=pc_dim,
        )

        if config.freeze_decoder:
            self.freeze_decoder()

        # map each source to its output head. Multiple sources can share a head.
        self.source_to_head = {
            "occupancy": "occupancy",
            "semantic_segmentation": "semantic_segmentation",
            "pseudo_occupancy": "occupancy",
            "pseudo_semantic_segmentation": "semantic_segmentation",
        }

        # map each source to its query group.
        self.source_to_query = {
            "occupancy": "occupancy",
            "semantic_segmentation": "occupancy",
            "pseudo_occupancy": "pseudo_occupancy",
            "pseudo_semantic_segmentation": "pseudo_occupancy",
        }

        # a list of which supervision tasks are enabled
        self.active_sources = get_active_sources(config)

        if semantic_segmentation_weights is not None:
            self.register_buffer("semantic_segmentation_weights", semantic_segmentation_weights)
        else:
            self.semantic_segmentation_weights = None

        self.min_query_time = min_query_time
        self.max_query_time = max_query_time

    @property
    def require_occupancy(self) -> bool:
        return self.config.occupancy

    @property
    def require_semantic_segmentation(self) -> bool:
        return self.config.semantic_segmentation

    @property
    def require_pseudo_occupancy(self) -> bool:
        return self.config.pseudo_occupancy

    @property
    def require_pseudo_semantic_segmentation(self) -> bool:
        return self.config.pseudo_semantic_segmentation

    def _build_decoder(
        self,
        input_dim: int,
        total_dims: int,
        point_cloud_range: PointCloudRange,
        gradient_checkpointing: bool,
        use_point_contraction: bool,
        point_contraction: Optional[PointContraction],
        pc_dim: int = 4,
    ) -> nn.Module:
        """Construct the inner per-query decoder; subclasses override to swap it out."""
        return UnoDecoder(
            in_feature_dim=input_dim,
            hidden_dim=self.config.hidden_dim,
            offset_dim=self.config.offset_dim,
            n_occ_resblocks=self.config.num_resblocks,
            num_offsets=self.config.num_offsets,
            output_dim=total_dims,
            pc_dim=pc_dim,
            point_cloud_range=point_cloud_range,
            gradient_checkpointing=gradient_checkpointing and self.config.decoder_checkpointing,
            tanh_offsets=self.config.tanh_offsets,
            tanh_offsets_scale=self.config.tanh_offsets_scale,
            coordinate_encoder_max_freq=self.config.coordinate_encoder_max_freq,
            use_point_contraction=use_point_contraction,
            point_contraction=point_contraction,
            chunking_by_pre_allocate=self.config.chunking_by_pre_allocate,
        )

    def freeze_decoder(self) -> None:
        for p in self._decoder.parameters():
            p.requires_grad = False

    def decoder(self, bev_feats: Tensor, queries: Tensor, max_queries: Optional[int] = None) -> dict:
        # Freezing is handled via requires_grad=False on the decoder parameters
        # (see freeze_decoder). Do not wrap in torch.no_grad() here: that would also
        # cut the gradient flow through bev_feats, silently detaching the encoder
        # from every loss computed through this decoder.
        if max_queries is None:
            return self._decoder(bev_feats, queries)
        return self._decoder.forward_chunks(bev_feats, queries, max_queries)

    def decode(
        self,
        bev_feats: Tensor,
        queries: Tensor,
        source_to_indices: Optional[Dict[str, slice]] = None,
        max_parallel_queries: Optional[int] = None,
    ) -> Dict[str, Tensor]:
        """
        Decode the queries.
        """
        if max_parallel_queries is None:
            max_parallel_queries = self.config.max_parallel_queries

        if queries.shape[-1] != self._decoder.pc_dim:
            queries = F.pad(queries, (0, self._decoder.pc_dim - queries.shape[-1]))

        if max_parallel_queries != -1 and queries.shape[1] > max_parallel_queries:
            out = self.decoder(bev_feats, queries, max_parallel_queries)
        else:
            out = self.decoder(bev_feats, queries)

        if source_to_indices is None:
            for head, dims in self.head_to_dims.items():
                logits = out["prediction"][..., dims]
                out[f"{head}_logits"] = logits
        else:
            pred = out["prediction"]
            for source in self.active_sources:
                dims = self.head_to_dims[self.source_to_head[source]]
                query_slice = source_to_indices[self.source_to_query[source]]
                out[f"{source}_logits"] = pred[:, query_slice, dims]

        return out

    def _get_queries(self, bev_feats: Tensor, inputs: Dict[str, dict]) -> Tuple[Tensor, dict]:
        """Concatenate the queries of all active query groups.

        Returns the concatenated queries and a mapping from query group name to the
        slice of its queries in the concatenated tensor.
        """
        # Activate a query group if any active source reads from it, so sources like
        # semantic segmentation (which share the occupancy queries) work even when the
        # occupancy loss itself is disabled.
        active_query_groups = {self.source_to_query[source] for source in self.active_sources}

        query_groups: List[Tuple[str, Tensor]] = []

        if "occupancy" in active_query_groups:
            query_groups.append(("occupancy", inputs["occupancy"]["occupancy_queries"]))

        if "pseudo_occupancy" in active_query_groups:
            query_groups.append(("pseudo_occupancy", inputs["pseudo_occupancy"]["occupancy_queries"]))

        # Pad all groups to a common last dim (e.g. queries lacking a z column)
        # so the concatenation below never fails, regardless of group order.
        max_query_dim = max((q.shape[-1] for _, q in query_groups), default=0)
        query_groups = [
            (name, F.pad(q, (0, max_query_dim - q.shape[-1])) if q.shape[-1] < max_query_dim else q)
            for name, q in query_groups
        ]

        source_to_query_indices = {}
        start_index = 0
        for source, queries in query_groups:
            num = queries.shape[1]
            source_to_query_indices[source] = slice(start_index, start_index + num)
            start_index += num

        if len(query_groups) == 0:
            active_queries = torch.empty(
                bev_feats.shape[0], 0, self._decoder.pc_dim, device=bev_feats.device, dtype=bev_feats.dtype
            )
        else:
            active_queries = torch.cat([queries for _, queries in query_groups], dim=1)

        return active_queries, source_to_query_indices

    def _get_total_loss(self, loss: dict) -> Tensor:
        total_loss = 0.0
        for key, value in loss.items():
            total_loss += value * getattr(self.config, f"{key}_weight")
        return total_loss

    def forward(
        self,
        bev_feats: Tensor,
        inputs: Dict[str, dict],
        label_mapper: Optional[LabelMapper] = None,
    ) -> Dict[str, Tensor]:
        """Decode all active query groups in one pass and compute the per-source losses.

        Args:
            bev_feats: The BEV feature map to decode from.
            inputs: Per-source input dicts keyed by source name. Recognized keys:
                "occupancy", "semantic_segmentation", "pseudo_occupancy",
                "pseudo_semantic_segmentation". Inactive sources may be omitted.
            label_mapper: Maps raw labels into the learning space; unused by the losses
                themselves, accepted for interface stability.
        """
        del label_mapper
        occupancy_inputs = inputs.get("occupancy", {})
        pseudo_occupancy_inputs = inputs.get("pseudo_occupancy", {})
        semantic_segmentation_inputs = inputs.get("semantic_segmentation", {})

        active_queries, source_to_query_indices = self._get_queries(bev_feats, inputs)

        if active_queries.shape[1] > 0:
            out = self.decode(bev_feats, active_queries, source_to_query_indices)
        else:
            out = {}

        # Pseudo points are data-dependent (camera depth + mask coverage) and can be empty for a
        # sample; fall back to a grad anchor so the step still produces gradients for every parameter.
        pseudo_queries = pseudo_occupancy_inputs.get("occupancy_queries")
        no_pseudo_queries = pseudo_queries is None or pseudo_queries.shape[1] == 0

        losses = {}
        if self.require_occupancy:
            losses["occupancy_loss"] = self._get_occupancy_loss(
                out["occupancy_logits"], occupancy_inputs["occupancy_label"]
            )

        if self.require_pseudo_occupancy:
            if no_pseudo_queries:
                losses["pseudo_occupancy_loss"] = self._empty_pseudo_grad_anchor(bev_feats)
            else:
                losses["pseudo_occupancy_loss"] = self._get_occupancy_loss(
                    out["pseudo_occupancy_logits"], pseudo_occupancy_inputs["occupancy_label"]
                ) * pseudo_occupancy_inputs.get("loss_weight", 1.0)

        if self.require_semantic_segmentation:
            losses["semantic_segmentation_loss"] = self._get_semantic_segmentation_loss(
                out["semantic_segmentation_logits"], semantic_segmentation_inputs
            )

        if self.require_pseudo_semantic_segmentation:
            if no_pseudo_queries:
                losses["pseudo_semantic_segmentation_loss"] = self._empty_pseudo_grad_anchor(bev_feats)
            else:
                losses["pseudo_semantic_segmentation_loss"] = self._get_semantic_segmentation_loss(
                    out["pseudo_semantic_segmentation_logits"], inputs["pseudo_semantic_segmentation"]
                ) * pseudo_occupancy_inputs.get("loss_weight", 1.0)

        if self.config.offsets_regularization and "offsets_coords" in out:
            losses["offsets_regularization_loss"] = self._get_offsets_regularization_loss(out["offsets_coords"])

        total_loss = self._get_total_loss(losses)
        out["one_head_losses"] = losses
        out["one_head_total_loss"] = total_loss
        out["source_to_indices"] = source_to_query_indices

        return out
