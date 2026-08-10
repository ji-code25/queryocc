"""QueryLossesMixin — per-query loss methods for the heads of OneHeadDecoder.

Occupancy (+pseudo), semantic cross-entropy (+pseudo), offsets regularization, and the
grad anchor that keeps DDP ranks in sync when a sample has no pseudo queries.
Mixed into OneHeadDecoder (decoder.py); holds METHODS ONLY — every parameter/buffer these
methods touch is created in decoder.py.__init__.
"""


import torch.nn.functional as F
from torch import Tensor


class QueryLossesMixin:
    def _get_occupancy_loss(self, occupancy_y: Tensor, occ_labels: Tensor) -> Tensor:
        """Binary cross-entropy occupancy loss."""
        return F.binary_cross_entropy_with_logits(occupancy_y, occ_labels.float(), reduction="mean")

    def _masked_semantic_cross_entropy(
        self, y_logits: Tensor, semantic_segmentation_inputs: dict, empty_loss: Tensor
    ) -> Tensor:
        """Cross entropy over valid labels only. Returns empty_loss when nothing is valid."""
        y_labels = semantic_segmentation_inputs["semantic_segmentation_labels"].flatten()
        y_valids = semantic_segmentation_inputs["semantic_segmentation_valid"].flatten() > 0

        if not self.config.supervise_free_class:
            y_valids = y_valids & (y_labels != -1)

        if y_valids.sum() == 0:
            return empty_loss

        y_logits = y_logits.flatten(start_dim=0, end_dim=1)
        y_logits = y_logits[y_valids, :]
        y_labels = y_labels[y_valids]

        if self.config.supervise_free_class:
            num_classes = y_logits.shape[1]
            y_labels[y_labels == -1] = num_classes - 1

        return F.cross_entropy(y_logits, y_labels, reduction="mean", weight=self.semantic_segmentation_weights)

    def _get_semantic_segmentation_loss(self, y_logits: Tensor, semantic_segmentation_inputs: dict) -> Tensor:
        # Graph-connected zero: keeps DDP ranks in sync when a sample has no valid labels
        # (e.g. all pseudo-labels filtered out by a high score threshold).
        empty_loss = y_logits.sum() * 0.0
        return self._masked_semantic_cross_entropy(y_logits, semantic_segmentation_inputs, empty_loss)

    def _get_offsets_regularization_loss(self, offsets_coords: Tensor) -> Tensor:
        """
        Get the offsets regularization loss.
        - We want to ensure that the offsets are not too large.
        - We want to ensure that the different offsets for one query are not too similar.
        offsets_coords: bs, num_queries, num_offsets, 2
        """

        norm_loss = offsets_coords.norm(dim=-1).mean()
        return norm_loss

    def _empty_pseudo_grad_anchor(self, bev_feats: Tensor) -> Tensor:
        """Zero-valued loss for samples without pseudo points; touches bev_feats and all decoder
        parameters so DDP gradients stay in sync when the pseudo queries are the only active group."""
        anchor = (bev_feats * 0.0).sum()
        for param in self._decoder.parameters():
            anchor = anchor + (param * 0.0).sum()
        return anchor
