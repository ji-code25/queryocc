import torch
from torch import Tensor
from torchmetrics import Accuracy, JaccardIndex

from queryocc.model.common.metrics import BinaryRayIoU, SemanticRayIoU
from queryocc.utils.fetchers import fetch_label_mapper


class PreTrainMetrics:
    """Mixin providing metric initialization and computation for PreTrainModel."""

    def _initialize_metrics(self):
        # Initialize occupancy metrics
        if self.net.config.use_one_head and self.net.config.one_head.occupancy:
            metrics = ["acc"]
            metric_classes = [Accuracy]

            for split in ["train", "val"]:
                for metric, cls in zip(metrics, metric_classes):
                    setattr(self, f"{split}_occupancy_{metric}", cls(task="binary"))

        if self.config.compute_occ3d_metrics:
            splits = ["train", "val", "test"] if self.config.compute_train_metrics else ["val", "test"]

            # Voxel-level binary metrics for lidar and camera masks across splits
            metrics = {
                "iou": JaccardIndex,
            }

            binary_masks = ["camera", "camera_dynamic", "camera_static", "none"]
            multiclass_masks = ["camera", "none"]

            # Probability sweep is per-split: validation uses a reduced set (the voxel-IoU/RayIoU
            # sweep is a large slice of val cost and scales with the threshold count), while
            # test/train keep the full benchmark sweep so reported numbers stay comparable.
            def prob_thresholds_for(split: str) -> list:
                cfg = self.config.occ3d_val_prob_thresholds if split == "val" else self.config.occ3d_prob_thresholds
                return list(cfg)

            def pstr(threshold) -> str:
                return str(threshold).replace(".", "p")

            # Initialize metrics
            for metric_name, metric_cls in metrics.items():
                for split in splits:
                    for mask in binary_masks:
                        for threshold in prob_thresholds_for(split):
                            self.add_module(
                                f"occ3d_metrics_binary_{mask}_{metric_name}_pthr{pstr(threshold)}_{split}",
                                metric_cls(task="binary", threshold=threshold),
                            )
                    if self.net.can_decode_semantic_segmentation:
                        # Add multiclass metrics
                        for mask in multiclass_masks:
                            self.add_module(
                                f"occ3d_metrics_multiclass_{mask}_{metric_name}_{split}",
                                metric_cls(
                                    task="multiclass",
                                    average="none",
                                    num_classes=len(self.net.config.semantic_segmentation_classes) + 2,
                                ),
                            )

            self.occ3d_metrics = list(metrics.keys())
            self.occ3d_metrics_binary_masks = binary_masks
            self.occ3d_metrics_multiclass_masks = multiclass_masks
            # Per-split string thresholds, consumed by the update driver and logging.
            self.occ3d_metrics_prob_thresholds = {
                split: [pstr(t) for t in prob_thresholds_for(split)] for split in splits
            }

            # Ray-level binary metrics
            ray_metrics = {
                "binary_rayiou": BinaryRayIoU,
            }
            depth_thresholds = [1, 2, 4]
            for split in splits:
                for depth_threshold in depth_thresholds:
                    for prob_threshold in prob_thresholds_for(split):
                        for metric_name, metric_cls in ray_metrics.items():
                            self.add_module(
                                f"occ3d_metrics_{metric_name}_pthr{pstr(prob_threshold)}_dthr{pstr(depth_threshold)}_{split}",
                                metric_cls(depth_threshold=depth_threshold, prob_threshold=prob_threshold),
                            )
            self.occ3d_ray_metrics = list(ray_metrics.keys())
            self.occ3d_ray_metrics_depth_thresholds = [pstr(t) for t in depth_thresholds]
            self.occ3d_ray_metrics_prob_thresholds = {
                split: [pstr(t) for t in prob_thresholds_for(split)] for split in splits
            }

            # Ray-level semantic metrics
            if self.net.can_decode_semantic_segmentation:
                ray_semantic_metrics = {
                    "semantic_rayiou": SemanticRayIoU,
                }
                for depth_threshold in depth_thresholds:
                    for metric_name, metric_cls in ray_semantic_metrics.items():
                        for split in splits:
                            self.add_module(
                                f"occ3d_metrics_{metric_name}_dthr{str(depth_threshold).replace('.', 'p')}_{split}",
                                metric_cls(
                                    depth_threshold=depth_threshold,
                                    class_names=self.net.config.semantic_segmentation_classes,
                                ),
                            )
                self.occ3d_ray_semantic_metrics = list(ray_semantic_metrics.keys())

    def _get_occ3d_mask(self, labels: Tensor, camera_masks: Tensor, lidar_masks: Tensor, mask_name: str) -> Tensor:
        label_mapper = fetch_label_mapper(self.trainer)
        if mask_name == "camera":
            mask = camera_masks
        elif mask_name == "lidar":
            mask = lidar_masks
        elif mask_name == "camera_dynamic":
            mask = camera_masks & label_mapper.learning_to_dynamic(labels)
        elif mask_name == "camera_static":
            mask = camera_masks & label_mapper.learning_to_static(labels)
        else:
            mask = torch.ones_like(labels)
        return mask.bool()

    def _compute_occ3d_metrics(self, batch: dict, res: dict, split: str) -> None:
        if "occ3d_binary_preds" not in res:
            return

        bs = res["occ3d_binary_preds"].shape[0]
        camera_masks = batch.get("occ3d_mask_camera").view(bs, -1)
        lidar_masks = batch.get("occ3d_mask_lidar").view(bs, -1)
        label_mapper = fetch_label_mapper(self.trainer)
        # get metrics with name "occ3d_metrics_binary_*"
        binary_probs = res["occ3d_binary_probs"].view(bs, -1)
        labels = batch["occ3d_labels"].view(bs, -1)
        binary_target = label_mapper.learning_to_binary_target(labels)
        if self.net.can_decode_semantic_segmentation:
            semantic_probs = res["occ3d_semantic_probs"].flatten(start_dim=0, end_dim=1)
            semantic_preds = res["occ3d_semantic_preds"].view(bs, -1)

        for metric_name in self.occ3d_metrics:
            for mask_name in self.occ3d_metrics_binary_masks:
                for prob_threshold in self.occ3d_metrics_prob_thresholds[split]:
                    mask = self._get_occ3d_mask(labels, camera_masks, lidar_masks, mask_name)
                    metric = getattr(
                        self, f"occ3d_metrics_binary_{mask_name}_{metric_name}_pthr{prob_threshold}_{split}"
                    )
                    metric.update(binary_probs[mask], binary_target[mask])

            if self.net.can_decode_semantic_segmentation:
                for mask_name in self.occ3d_metrics_multiclass_masks:
                    mask = self._get_occ3d_mask(labels, camera_masks, lidar_masks, mask_name)
                    metric = getattr(self, f"occ3d_metrics_multiclass_{mask_name}_{metric_name}_{split}")
                    metric.update(semantic_probs[mask.flatten(), :], labels[mask])

        # Ray-level occ3d metrics: DVR raycasting dominates validation cost and is largely redundant
        # across the threshold sweep. The GT raycast is threshold-independent and the prediction
        # raycast depends only on prob_threshold; depth_threshold affects only the cheap L1 compare in
        # _calc_rayiou. Raycast once and fan the result out to every (prob, depth) instance instead of
        # re-raycasting per instance (~48 -> ~10 DVR calls/sample). Numerically identical to per-metric
        # update() (which is kept for any external callers).
        for metric_name in self.occ3d_ray_metrics:
            self._update_binary_rayiou_dedup(
                metric_name, binary_probs, binary_target, batch["lidar_origins"], batch["sample_token"], split
            )

        if self.net.can_decode_semantic_segmentation:
            for metric_name in self.occ3d_ray_semantic_metrics:
                self._update_semantic_rayiou_dedup(
                    metric_name, semantic_preds, labels, batch["lidar_origins"], batch["sample_token"], split
                )

    @torch.no_grad()
    def _update_binary_rayiou_dedup(
        self,
        metric_name: str,
        binary_probs: Tensor,
        binary_target: Tensor,
        lidar_origins: Tensor,
        sample_token: list,
        split: str,
    ) -> None:
        p_strs = self.occ3d_ray_metrics_prob_thresholds[split]
        d_strs = self.occ3d_ray_metrics_depth_thresholds
        # Every (prob, depth) instance shares identical raycasting buffers; borrow one to raycast with.
        ref = getattr(self, f"occ3d_metrics_{metric_name}_pthr{p_strs[0]}_dthr{d_strs[0]}_{split}")
        prob_vals = [
            getattr(self, f"occ3d_metrics_{metric_name}_pthr{p}_dthr{d_strs[0]}_{split}").prob_threshold for p in p_strs
        ]
        for i in range(binary_probs.shape[0]):
            if sample_token[i] == "n/a":
                continue
            origin = lidar_origins[i].unsqueeze(0)
            pcd_gt = ref._process_one_sample(binary_target[i].reshape(200, 200, 16), origin)
            valid = pcd_gt[:, 0].int() != 0  # evaluate on non-free rays (matches _update_one_batch)
            pcd_gt = pcd_gt[valid]
            for p, prob_val in zip(p_strs, prob_vals):
                preds = (binary_probs[i] > prob_val).float().reshape(200, 200, 16)
                pcd_pred = ref._process_one_sample(preds, origin)[valid]
                for d in d_strs:
                    getattr(self, f"occ3d_metrics_{metric_name}_pthr{p}_dthr{d}_{split}")._calc_rayiou(pcd_pred, pcd_gt)

    @torch.no_grad()
    def _update_semantic_rayiou_dedup(
        self,
        metric_name: str,
        semantic_preds: Tensor,
        labels: Tensor,
        lidar_origins: Tensor,
        sample_token: list,
        split: str,
    ) -> None:
        d_strs = self.occ3d_ray_metrics_depth_thresholds
        # Semantic pred/GT raycasts are both depth-threshold-independent: raycast once, reuse for all depths.
        ref = getattr(self, f"occ3d_metrics_{metric_name}_dthr{d_strs[0]}_{split}")
        free_id = len(ref.class_names) - 1
        for i in range(semantic_preds.shape[0]):
            if sample_token[i] == "n/a":
                continue
            origin = lidar_origins[i].unsqueeze(0)
            pcd_pred = ref._process_one_sample(semantic_preds[i].reshape(200, 200, 16), origin)
            pcd_gt = ref._process_one_sample(labels[i].reshape(200, 200, 16), origin)
            valid = pcd_gt[:, 0].int() != free_id  # evaluate on non-free rays
            pcd_pred, pcd_gt = pcd_pred[valid], pcd_gt[valid]
            for d in d_strs:
                getattr(self, f"occ3d_metrics_{metric_name}_dthr{d}_{split}")._calc_rayiou(pcd_pred, pcd_gt)
