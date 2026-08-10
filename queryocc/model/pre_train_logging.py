import numpy as np
import torch

from queryocc.utils.fetchers import fetch_label_mapper


class PreTrainLogging:
    """Mixin providing training/validation logging for PreTrainModel."""

    def _log_train(self, batch: dict, res: dict):
        # log losses
        self.log(
            "train/total_loss", res["loss"].detach(), on_step=True, rank_zero_only=self.config.rank_zero_only_logging
        )
        for k, v in res.items():
            if k.endswith("_loss"):
                self.log(
                    f"train/loss/{k[:-5]}", v.detach(), on_step=True, rank_zero_only=self.config.rank_zero_only_logging
                )

    def _log_stuff_epoch(self, split: str) -> None:
        label_mapper = fetch_label_mapper(self.trainer)
        if self.config.compute_occ3d_metrics and self._compute_metrics:
            for metric_name in self.occ3d_metrics:
                for mask_name in self.occ3d_metrics_binary_masks:
                    values_for_max = []
                    for prob_threshold in self.occ3d_metrics_prob_thresholds[split]:
                        metric = getattr(
                            self,
                            f"occ3d_metrics_binary_{mask_name}_{metric_name}_pthr{prob_threshold}_{split}",
                        )
                        metric_value = metric.compute()
                        if self.config.log_individual_occ3d_metrics:
                            self.log(
                                f"{split}/occ3d/metrics/occ3d_binary_iou/{mask_name}/{metric_name}_pthr{prob_threshold}",
                                metric_value,
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                                rank_zero_only=self.config.rank_zero_only_logging,
                            )
                        values_for_max.append(metric_value)
                        metric.reset()
                    values_for_max = torch.stack(values_for_max)
                    max_value = values_for_max.max()
                    self.log(
                        f"{split}/occ3d/metrics/occ3d_binary_iou/{mask_name}/max_{metric_name}",
                        max_value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        rank_zero_only=self.config.rank_zero_only_logging,
                    )

                if self.net.can_decode_semantic_segmentation and label_mapper is not None:
                    # Reduced learning spaces don't necessarily contain every occ3d_mean_classes
                    # entry — average over the intersection instead of crashing.
                    mean_class_names = [
                        c for c in self.config.occ3d_mean_classes if c in label_mapper._learning_label_to_idx
                    ]
                    mean_indices = label_mapper.learning_label_to_idx(mean_class_names)
                    mean_indices = np.array(mean_indices)
                    dynamic_indices = mean_indices[label_mapper.learning_to_dynamic(mean_indices)]
                    static_indices = mean_indices[label_mapper.learning_to_static(mean_indices)]
                    mean_indices = torch.tensor(mean_indices, device=self.device)
                    dynamic_indices = torch.tensor(dynamic_indices, device=self.device)
                    static_indices = torch.tensor(static_indices, device=self.device)

                    semantic_segmentation_class_indices = label_mapper.learning_label_to_idx(
                        self.net.config.semantic_segmentation_classes
                    )
                    semantic_segmentation_class_indices = torch.tensor(
                        semantic_segmentation_class_indices, device=self.device
                    )

                    for mask_name in self.occ3d_metrics_multiclass_masks:
                        metric = getattr(self, f"occ3d_metrics_multiclass_{mask_name}_{metric_name}_{split}")
                        metric_value = metric.compute()
                        for class_idx, class_name in zip(
                            semantic_segmentation_class_indices, self.net.config.semantic_segmentation_classes
                        ):
                            self.log(
                                f"{split}/occ3d/metrics/occ3d_multiclass_iou/{mask_name}/{metric_name}/z_{class_name}",
                                metric_value[class_idx],
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                                rank_zero_only=self.config.rank_zero_only_logging,
                            )
                        metric.reset()

                        mean = metric_value[mean_indices].mean()
                        self.log(
                            f"{split}/occ3d/metrics/occ3d_multiclass_iou/{mask_name}/{metric_name}/mean",
                            mean,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                            rank_zero_only=self.config.rank_zero_only_logging,
                        )

                        mean_dynamic = metric_value[dynamic_indices].mean()
                        self.log(
                            f"{split}/occ3d/metrics/occ3d_multiclass_iou/{mask_name}/{metric_name}/mean_dynamic",
                            mean_dynamic,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                            rank_zero_only=self.config.rank_zero_only_logging,
                        )

                        mean_static = metric_value[static_indices].mean()
                        self.log(
                            f"{split}/occ3d/metrics/occ3d_multiclass_iou/{mask_name}/{metric_name}/mean_static",
                            mean_static,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                            rank_zero_only=self.config.rank_zero_only_logging,
                        )

            for metric_name in self.occ3d_ray_metrics:
                values_for_max = []
                for prob_threshold in self.occ3d_ray_metrics_prob_thresholds[split]:
                    values_for_mean = []
                    for depth_threshold in self.occ3d_ray_metrics_depth_thresholds:
                        metric = getattr(
                            self, f"occ3d_metrics_{metric_name}_pthr{prob_threshold}_dthr{depth_threshold}_{split}"
                        )
                        metric_value = metric.compute()
                        # Log individual metric values
                        if self.config.log_individual_occ3d_metrics:
                            self.log(
                                f"{split}/occ3d/metrics/occ3d_binary_rayiou/{metric_name}_pthr{prob_threshold}_dthr{depth_threshold}",
                                metric_value,
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                                rank_zero_only=self.config.rank_zero_only_logging,
                            )
                        values_for_mean.append(metric_value)
                        metric.reset()

                    # Compute mean across all processes
                    values_for_mean = torch.stack(values_for_mean)
                    mean_value = values_for_mean.mean()
                    values_for_max.append(mean_value)

                    # Log mean value
                    self.log(
                        f"{split}/occ3d/metrics/occ3d_binary_rayiou/mean_{metric_name}_pthr{prob_threshold}",
                        mean_value,
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        rank_zero_only=self.config.rank_zero_only_logging,
                    )

                # Compute max across all processes
                values_for_max = torch.stack(values_for_max)
                max_value = values_for_max.max()

                # Log max value
                self.log(
                    f"{split}/occ3d/metrics/occ3d_binary_rayiou/max_{metric_name}",
                    max_value,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

            if self.net.can_decode_semantic_segmentation:
                # Reduced learning spaces don't necessarily contain every occ3d_mean_classes
                # entry — average over the intersection instead of crashing.
                mean_class_names = [
                    c for c in self.config.occ3d_mean_classes if c in label_mapper._learning_label_to_idx
                ]
                mean_indices = label_mapper.learning_label_to_idx(mean_class_names)
                mean_indices = np.array(mean_indices)
                dynamic_mask = label_mapper.learning_to_dynamic(mean_indices)
                static_mask = label_mapper.learning_to_static(mean_indices)
                mean_indices = torch.tensor(mean_indices, device=self.device)
                dynamic_mask = torch.tensor(dynamic_mask, device=self.device)
                static_mask = torch.tensor(static_mask, device=self.device)
                for metric_name in self.occ3d_ray_semantic_metrics:
                    num_depth_thresholds = len(self.occ3d_ray_metrics_depth_thresholds)
                    num_semantic_classes = len(mean_class_names)
                    all_metric_values = torch.zeros(num_depth_thresholds, num_semantic_classes, device=self.device)

                    for i, depth_threshold in enumerate(self.occ3d_ray_metrics_depth_thresholds):
                        metric = getattr(self, f"occ3d_metrics_{metric_name}_dthr{depth_threshold}_{split}")
                        metric_values = metric.compute()  # Shape: [1 + num_classes + 1] (includes "noise" and "free")
                        mean_metric_values = metric_values[mean_indices]  # Extract only the mean classes
                        mean_per_depth = mean_metric_values.mean()
                        all_metric_values[i, :] = mean_metric_values

                        # Log mean per depth
                        self.log(
                            f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}_dthr{depth_threshold}/mean",
                            mean_per_depth,
                            on_step=False,
                            on_epoch=True,
                            sync_dist=True,
                            rank_zero_only=self.config.rank_zero_only_logging,
                        )

                        # Log individual class metrics
                        for j, class_name in enumerate(mean_class_names):
                            self.log(
                                f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}_dthr{depth_threshold}/{class_name}",
                                mean_metric_values[j],
                                on_step=False,
                                on_epoch=True,
                                sync_dist=True,
                                rank_zero_only=self.config.rank_zero_only_logging,
                            )
                        metric.reset()

                # Calculate overall means
                mean_per_class = all_metric_values.mean(dim=0)  # Mean across depth thresholds
                overall_mean = all_metric_values.mean()  # Overall mean
                dynamic_mean = all_metric_values[:, dynamic_mask].mean()
                static_mean = all_metric_values[:, static_mask].mean()

                # Log class-wise means across all depth thresholds
                for j, class_name in enumerate(mean_class_names):
                    self.log(
                        f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}_mean_depths/{class_name}",
                        mean_per_class[j],
                        on_step=False,
                        on_epoch=True,
                        sync_dist=True,
                        rank_zero_only=self.config.rank_zero_only_logging,
                    )

                # Log overall mean
                self.log(
                    f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}/overall_mean",
                    overall_mean,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

                self.log(
                    f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}/overall_mean_dynamic",
                    dynamic_mean,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
                self.log(
                    f"{split}/occ3d/metrics/occ3d_sem_rayiou/{metric_name}/overall_mean_static",
                    static_mean,
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

    def on_validation_epoch_end(self) -> None:
        self._log_stuff_epoch("val")

    def on_train_epoch_end(self) -> None:
        # Train-split metrics (occ3d mIoU/RayIoU, ...) are accumulated every training_step when
        # compute_train_metrics is set, but had no epoch-end hook to compute+log them (only
        # val/test did) -- so they sat there silently. Surface them so an overfit sanity check can
        # read a TRAIN occ3d number (does it actually fit?), not just the held-out val one.
        # Gated on compute_train_metrics -> zero cost for normal full runs (which leave it off).
        if self.config.compute_train_metrics:
            self._log_stuff_epoch("train")

    def on_test_epoch_end(self) -> None:
        self._log_stuff_epoch("test")

    def _log_val(self, batch: dict, res: dict):
        # log losses
        self.log(
            "val/total_loss",
            res["loss"].detach(),
            on_step=False,
            on_epoch=True,
            batch_size=res["batch_size"],
            sync_dist=True,
            rank_zero_only=self.config.rank_zero_only_logging,
        )
        for k, v in res.items():
            if k.endswith("_loss"):
                self.log(
                    f"val/loss/{k[:-5]}",
                    v.detach(),
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )

        # Log task metrics
        if "occupancy_predictions" in res:
            if "occupancy_label" in batch:
                occupancy_label = batch["occupancy_label"]
            elif "occupancy_label" in res:
                occupancy_label = res["occupancy_label"]
            else:
                occupancy_label = None

            if hasattr(self, "val_occupancy_acc"):
                self.val_occupancy_acc(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/accuracy",
                    self.val_occupancy_acc,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_prec"):
                self.val_occupancy_prec(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/precision",
                    self.val_occupancy_prec,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_rec"):
                self.val_occupancy_rec(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/recall",
                    self.val_occupancy_rec,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
            if hasattr(self, "val_occupancy_f1"):
                self.val_occupancy_f1(res["occupancy_predictions"], occupancy_label)
                self.log(
                    "val/occupancy/f1",
                    self.val_occupancy_f1,
                    batch_size=res["batch_size"],
                    on_step=False,
                    on_epoch=True,
                    sync_dist=True,
                    rank_zero_only=self.config.rank_zero_only_logging,
                )
