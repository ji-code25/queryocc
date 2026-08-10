import logging
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from queryocc.data.datasets.zod_base import ZODBaseDatasetConfig
from queryocc.data.occupancy.occupancy_sampler_gpu import LidarGPUOccupancySamplerConfig
from queryocc.model.base import BaseModel, ModelConfig
from queryocc.model.components.decoders.one_head import OneHeadDecoder
from queryocc.model.pre_train_logging import PreTrainLogging
from queryocc.model.pre_train_metrics import PreTrainMetrics
from queryocc.model.pre_train_probe import PreTrainProbe
from queryocc.utils.fetchers import fetch_label_mapper

# Suppress the specific deprecation warning about torch.cpu.amp.autocast
warnings.filterwarnings("ignore", message=r".*torch\.cpu\.amp\.autocast.*is deprecated.*")

logger = logging.getLogger(__name__)

TOLERATED_MISSING_KEYS = ("probe_vfm", "initial_features")
"""State-dict keys a checkpoint may omit without the load counting as incomplete.

``probe_vfm``: the optional post-train probe head, absent from runs that did not train one.
``initial_features``: the lifter's constant BEV seed grid (zeros or a fixed sinusoidal encoding),
a persistent buffer rebuilt identically at construction. Released checkpoints omit it —
``tools/release/export_weights.py`` drops only entries a freshly built network reproduces
bit-exactly, and verifies the result against this list."""


@dataclass
class PreTrainConfig(ModelConfig):
    _target: type = field(default_factory=lambda: PreTrainModel)

    freeze_encoder: bool = False
    """ Whether to freeze the encoder or not """

    weight_img_depth_loss: float = 0.01
    """ Weight of the image depth loss """
    weight_one_head_total_loss: float = 1.0
    """ Weight of the one head total loss """

    occupancy_sampler: LidarGPUOccupancySamplerConfig = field(default_factory=LidarGPUOccupancySamplerConfig)
    """ The occupancy sampler configuration """

    compute_occ3d_metrics: bool = False
    """ Whether to compute occ3d metrics or not """
    probe_occ3d_setting: Literal[1, 2] = 1
    """ The setting to use for the probe occ3d. 1: probe at the voxel centers, 2: probe at the
    voxel corners. """

    log_individual_occ3d_metrics: bool = False
    """ Whether to log individual occ3d metrics or not """
    occ3d_prob_thresholds: Tuple[float, ...] = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    """ Probability thresholds swept for occ3d voxel-IoU and RayIoU. The reported metric is the max
    over these, so this is the test/benchmark sweep - keep it full for comparable numbers. """
    occ3d_val_prob_thresholds: Tuple[float, ...] = (0.4, 0.6, 0.8)
    """ Reduced sweep used during *training-time* validation only. The voxel-IoU sweep dominates the
    val metric cost (~17% of the val step, scaling with the threshold count); a coarser sweep keeps a
    good proxy of the max while cutting that cost. Test/train splits still use occ3d_prob_thresholds.
    Set equal to occ3d_prob_thresholds to disable. """
    val_num_query_points: Optional[int] = 40_000
    """ Number of query points the occupancy / pseudo-occupancy sampler draws during *validation*
    (instead of occupancy_sampler.num_query_points). _step_one_head's sampled-query decode +
    pseudo-occ/semantic loss is ~30% of the val step but only feeds the val/loss curves - the occ3d
    IoU/RayIoU come from the separate dense probe - so a smaller val budget keeps a loss monitor at a
    fraction of the cost. Default 40k trades a little extra cost (vs 20k) for a steadier val loss;
    set to None to use the full training count. """
    occ3d_mean_classes: Tuple[str, ...] = (
        "barrier",
        "bicycle",
        "bus",
        "car",
        "construction_vehicle",
        "motorcycle",
        "pedestrian",
        "traffic_cone",
        "trailer",
        "truck",
        "driveable_surface",
        "sidewalk",
        "terrain",
        "manmade",
        "vegetation",
    )
    """ The classes to use for mean occ3d metrics """

    rank_zero_only_logging: bool = True
    """ Whether to log only for rank zero or not """

    def __post_init__(self) -> None:
        super().__post_init__()

    def configure_dataset(self, dataset: ZODBaseDatasetConfig):  # pragma: no cover
        super().configure_dataset(dataset)
        dataset.load_offset_pointclouds = self.net.need_offset_pointclouds

        if self.net.need_vfm_stuff:
            dataset.load_vfm_stuff = True

        if self.net.cerveza_encoder.lss.supervise_depth:
            dataset.load_depth_maps = True

        if self.compute_occ3d_metrics:
            dataset.load_occ3d_val = True
        if self.compute_train_metrics and self.compute_occ3d_metrics:
            dataset.load_occ3d_train = True

        if (self.net.use_one_head and self.net.one_head.semantic_segmentation) or (
            "semantic_labels" in self.occupancy_sampler.positive_samples_based_on
        ):
            dataset.panoptic_labels_classes = self.net.semantic_segmentation_classes
            dataset.load_offset_panoptic_labels = True

        if self.net.use_one_head:
            if self.net.one_head.need_vfm_stuff_depth:
                dataset.vfm_cfg.read_metric3d_depth = True

            if self.net.one_head.need_vfm_stuff_semantics:
                dataset.vfm_cfg.read_grounding_sam = True

            if self.net.one_head.pseudo_semantic_segmentation:
                dataset.vfm_cfg.create_pseudo_pointclouds = True

        self._min_query_time = -dataset.max_past_frame_s if dataset.num_past_frames_to_sample > 0 else -0.1
        self._max_query_time = dataset.max_future_frame_s if dataset.num_future_frames_to_sample > 0 else 0.1


class PreTrainModel(PreTrainProbe, PreTrainMetrics, PreTrainLogging, BaseModel):
    config: PreTrainConfig

    def __init__(self, config: PreTrainConfig, visualization_mode: bool = False):
        super().__init__(config, visualization_mode=visualization_mode)

        if self.net.config.f_use_occupancy_sampler:
            self.occupancy_sampler = self.config.occupancy_sampler.setup(
                point_cloud_range=self.config.point_cloud_range
            )

        if config.freeze_encoder:
            self.freeze_encoder()

        self._initialize_metrics()

    def freeze_encoder(self) -> None:
        for enc in self.net.get_encoders():
            for p in enc.parameters():
                p.requires_grad = False

    @property
    def _do_occ3d_step(self):
        return self.config.compute_occ3d_metrics and self._compute_metrics

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Path,
        config: PreTrainConfig,
        strict=True,
        is_from_pretrain: bool = True,
        visualization_mode: bool = False,
    ) -> "PreTrainModel":
        """Load the model from a checkpoint."""
        # This code assumes the checkpoint is from a Pretrained model, if we want to load from a BEVSegmentor model,
        # we should use the resume training flag
        del is_from_pretrain  # we dont use this here
        model: PreTrainModel = config.setup(visualization_mode=visualization_mode)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        sd = checkpoint["state_dict"]

        # "dos." is the pre-rename prefix of the core network in older checkpoints.
        sdrep = {k.split(".", 1)[1]: v for k, v in sd.items() if k.startswith(("net.", "dos."))}
        sdrep = {k.replace("one_head_decoder.decoder", "one_head_decoder._decoder"): v for k, v in sdrep.items()}
        sdrep = {k: v for k, v in sdrep.items() if not k.startswith("one_head_decoder.rendering.decoder")}

        okay_to_miss = TOLERATED_MISSING_KEYS

        info = model.net.load_state_dict(sdrep, strict=False)
        missing_keys = [mk for mk in info.missing_keys if not any(ok in mk for ok in okay_to_miss)]
        unexpected_keys = list(info.unexpected_keys)

        if missing_keys or unexpected_keys:
            summary = (
                f"{len(missing_keys)} missing key(s) and {len(unexpected_keys)} unexpected key(s) "
                f"when loading {checkpoint_path}. Missing: {missing_keys[:10]}"
                f"{' ...' if len(missing_keys) > 10 else ''}. Unexpected: {unexpected_keys[:10]}"
                f"{' ...' if len(unexpected_keys) > 10 else ''}."
            )
            if strict:
                raise ValueError(
                    summary + " The checkpoint does not match this preset's architecture — check that the "
                    "preset matches the one it was trained with. Pass --allow_partial_load if the "
                    "mismatch is intended."
                )
            # Never silent: missing keys stay at their random initialization, which is
            # indistinguishable from a trained model until the metrics come out as noise.
            logger.warning("Partial weight load: missing parameters keep their initialization. %s", summary)

        return model

    def _encode(self, batch: dict) -> Tuple[Tensor, dict]:
        lidar = self._get_lidar_data(batch)
        camera = self._get_camera_data(batch)
        return self.net.encode(lidar, camera)  # type: ignore

    def step(self, batch: dict, batch_idx: int) -> dict:
        bs = len(batch["id"])

        if self.config.freeze_encoder:
            with torch.no_grad():
                bev_feats, info = self._encode(batch)
        else:
            bev_feats, info = self._encode(batch)

        res = {
            "batch_size": bs,
            "loss": torch.zeros(1, device=self.device, dtype=bev_feats.dtype),
            "info": info,
        }

        if self.config.logging.bev_features:
            res["info"]["bev_features"] = bev_feats

        if (
            self.config.net.image_encoder_name == "cerveza"
            and self.config.net.cerveza_encoder.lss.supervise_depth
            and "depth_maps" in batch
        ):
            img_depths = info["img_depths"][:, 0, ...]  # Currently only using the current image depth
            res["img_depths"] = img_depths
            res = self._step_img_depth(batch, img_depths, res)

        if self._do_occ3d_step:
            res = self._step_occ3d(batch, bev_feats, res)

        if self.net.config.use_one_head:
            res = self._step_one_head(batch, bev_feats, res)

        res = detach_dict(res, skip_keys=["loss"])

        return res

    def _prepare_occupancy_sampler_inputs(self, batch: dict) -> Tuple[dict, dict, dict, dict]:
        """Run occupancy and pseudo-occupancy samplers and package their outputs.

        Returns:
            occupancy_inputs: decoder input dict, or {} if sampling was skipped
            aux_labels: per-point labels (semantics, etc.), or {}
            pseudo_occupancy_inputs: pseudo decoder input dict, or {} if skipped
            pseudo_occ_aux_labels: per-point labels for pseudo points, or {}
        """
        occupancy_inputs: dict = {}
        aux_labels: dict = {}
        pseudo_occupancy_inputs: dict = {}
        pseudo_occ_aux_labels: dict = {}

        needs_occupancy = (
            self.net.one_head_decoder.require_occupancy or self.net.one_head_decoder.require_semantic_segmentation
        )
        if needs_occupancy:
            positive_point_labels = batch.get("offset_pointclouds_labels", {})
            occ_queries, occ_labels, aux_labels = self.occupancy_sampler(
                positive_points=batch["offset_pointclouds"],
                ego_poses=batch["ego_poses"],
                ego_times=batch["ego_pose_times"],
                extrinsics=batch["lidar_extrinsics"],
                negative_rays=batch.get("missing_rays", None),
                future_points_ratio_inside=batch["future_points_ratio_inside"],
                positive_point_labels=positive_point_labels,
                label_mapper=fetch_label_mapper(self.trainer),
            )
            occupancy_inputs = {"occupancy_queries": occ_queries, "occupancy_label": occ_labels}

        if self.net.one_head_decoder.require_pseudo_occupancy:
            pseudo_pointcloud_labels = {
                "panoptic_lidarseg": batch.get("pseudo_pointcloud_labels", None),
                "panoptic_valid": batch.get("pseudo_pointcloud_valids", None),
            }

            pseudo_extrinsics = batch["pseudo_pointcloud_extrinsics"]
            # (B, T, 4, 4) when cameras vary per timestep: select extrinsics per point via the timestep indices
            per_timestep_extrinsics = pseudo_extrinsics.dim() == 4
            pseudo_occ_queries, pseudo_occ_labels, pseudo_occ_aux_labels = self.occupancy_sampler(
                positive_points=batch["pseudo_pointcloud"],
                ego_poses=batch["pseudo_pointcloud_ego_poses"],
                ego_times=batch["pseudo_pointcloud_ego_timestamps"],
                extrinsics=pseudo_extrinsics,
                negative_rays=batch.get("pseudo_missing_rays", None),
                future_points_ratio_inside=batch["pseudo_points_ratio_inside"],
                positive_point_labels=pseudo_pointcloud_labels,
                label_mapper=fetch_label_mapper(self.trainer),
                positive_t_idx=batch["pseudo_pointcloud_t_idx"] if per_timestep_extrinsics else None,
                negative_rays_t_idx=batch["pseudo_missing_rays_t_idx"] if per_timestep_extrinsics else None,
            )
            pseudo_occupancy_inputs = {
                "occupancy_queries": pseudo_occ_queries,
                "occupancy_label": pseudo_occ_labels,
                "loss_weight": batch.get("pseudo_supervision_loss_weight", 1.0),
            }

        return occupancy_inputs, aux_labels, pseudo_occupancy_inputs, pseudo_occ_aux_labels

    def _process_one_head_outputs(self, one_head_out: dict, res: dict) -> dict:
        """Unpack decoder outputs into res and compute derived predictions."""
        for key, value in one_head_out.items():
            if "loss" in key:
                continue
            if key in res:
                raise ValueError(f"Key {key} already in res, please check if you are not overwriting a key")
            res[key] = value

        for key, value in one_head_out["one_head_losses"].items():
            res[f"one_head_{key}"] = value.detach()

        if "occupancy_logits" in res:
            res["occupancy_probs"] = res["occupancy_logits"].sigmoid()
            res["occupancy_predictions"] = (res["occupancy_logits"] > 0.0).float()

        if "semantic_segmentation_logits" in res:
            res["semantic_segmentation_predictions"] = res["semantic_segmentation_logits"].argmax(dim=-1)

        if "pseudo_semantic_segmentation_logits" in res:
            res["pseudo_semantic_segmentation_predictions"] = res["pseudo_semantic_segmentation_logits"].argmax(dim=-1)

        return res

    @contextmanager
    def _val_query_budget(self):
        """Temporarily shrink the occupancy sampler's query count during validation
        (config.val_num_query_points). No-op in training, when unset, or with no sampler."""
        cfg = getattr(getattr(self, "occupancy_sampler", None), "config", None)
        override = (not self.training) and self.config.val_num_query_points is not None and cfg is not None
        if override:
            original = cfg.num_query_points
            cfg.num_query_points = self.config.val_num_query_points
        try:
            yield
        finally:
            if override:
                cfg.num_query_points = original

    def _step_one_head(self, batch: dict, bev_feats: Tensor, res: dict) -> dict:
        decoder = self.net.one_head_decoder
        assert isinstance(decoder, OneHeadDecoder)

        with self._val_query_budget():
            (
                occupancy_inputs,
                aux_labels,
                pseudo_occupancy_inputs,
                pseudo_occ_aux_labels,
            ) = self._prepare_occupancy_sampler_inputs(batch)

        if occupancy_inputs:
            res["occupancy_queries"] = occupancy_inputs["occupancy_queries"]
            res["occupancy_label"] = occupancy_inputs["occupancy_label"]
            res["occupancy_aux_labels"] = aux_labels
        if pseudo_occupancy_inputs:
            res["pseudo_occupancy_queries"] = pseudo_occupancy_inputs["occupancy_queries"]
            res["pseudo_occupancy_label"] = pseudo_occupancy_inputs["occupancy_label"]
            res["pseudo_occupancy_aux_labels"] = pseudo_occ_aux_labels

        inputs: Dict[str, dict] = {
            "occupancy": occupancy_inputs,
            "pseudo_occupancy": pseudo_occupancy_inputs,
        }

        if decoder.require_semantic_segmentation:
            inputs["semantic_segmentation"] = {
                "semantic_segmentation_labels": aux_labels["panoptic_lidarseg"],
                "semantic_segmentation_valid": aux_labels["panoptic_valid"],
            }
        if decoder.require_pseudo_semantic_segmentation:
            inputs["pseudo_semantic_segmentation"] = {
                "semantic_segmentation_labels": pseudo_occ_aux_labels["panoptic_lidarseg"],
                "semantic_segmentation_valid": pseudo_occ_aux_labels["panoptic_valid"],
            }

        one_head_out = decoder(bev_feats, inputs, label_mapper=fetch_label_mapper(self.trainer))

        res["loss"] += one_head_out["one_head_total_loss"] * self.config.weight_one_head_total_loss

        return self._process_one_head_outputs(one_head_out, res)

    def _step_occ3d(self, batch: dict, bev_feats: Tensor, res: dict) -> dict:
        occ3d_labels = batch.get("occ3d_labels", None)

        if occ3d_labels is None:
            return res

        occ3d_out = self.probe_occ3d(bev_feats)

        if self.net.can_decode_semantic_segmentation:
            res["occ3d_semantic_logits"] = occ3d_out["semantic_segmentation_logits"].detach()
            res["occ3d_semantic_probs"] = occ3d_out["semantic_segmentation_probs"].detach()
            res["occ3d_semantic_preds"] = occ3d_out["semantic_segmentation_preds"].detach()

        if self.net.can_decode_occupancy:
            res["occ3d_binary_logits"] = occ3d_out["occupancy"].detach()
            res["occ3d_binary_probs"] = occ3d_out["occupancy"].detach().sigmoid()
            res["occ3d_binary_preds"] = (occ3d_out["occupancy"] > 0).detach().int()
        else:
            binary_probs = 1 - res["occ3d_semantic_probs"][..., -1]
            binary_pred = res["occ3d_semantic_preds"] != (occ3d_out["semantic_segmentation_logits"].shape[-1] - 1)
            res["occ3d_binary_probs"] = binary_probs.detach()
            res["occ3d_binary_preds"] = binary_pred.detach().int()

        return res

    def _step_img_depth(self, batch: dict, img_depths: Tensor, res: dict) -> dict:
        target = torch.stack(list(batch["depth_maps"].values()), dim=1)
        valids = torch.stack(list(batch["depth_maps_valid"].values()), dim=1)

        available_depths = self.net.image_encoder.lifter.depths
        min_depth = available_depths[0]
        max_depth = available_depths[-1]

        # Update valids to be between min and max dpeth
        valids = torch.logical_and(valids, target > min_depth)
        valids = torch.logical_and(valids, target < max_depth)
        valids = valids.float()

        # Find the closest possible bin (broadcast against [D] instead of repeating the bins)
        target_bins = torch.argmin(torch.abs(target[..., None] - available_depths), dim=-1)

        img_depths = img_depths.permute(0, 4, 1, 2, 3)

        loss = F.cross_entropy(
            img_depths,
            target_bins,
            reduction="none",
        )

        loss = loss * valids
        loss = loss.sum() / (valids.sum() + 1e-6)
        res["loss"] += self.config.weight_img_depth_loss * loss
        res["img_depth_loss"] = loss.detach()
        res["img_depth"] = img_depths.detach().argmax(1)
        res["target_depth"] = target_bins.detach()
        res["valid_depth"] = valids.detach()

        return res

    def training_step(self, batch: dict, batch_idx: int) -> Optional[dict]:
        if self.net.one_head_decoder.require_pseudo_occupancy and "pseudo_pointcloud" in batch:
            local_valid = batch["pseudo_pointcloud"].new_tensor(float(batch["pseudo_pointcloud"].shape[1] > 0))
            valid_ranks = local_valid.clone()
            world_size = 1
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                torch.distributed.all_reduce(valid_ranks, op=torch.distributed.ReduceOp.SUM)
                world_size = torch.distributed.get_world_size()

            if valid_ranks.item() == 0:
                if self.global_rank == 0:
                    print("WARNING: All ranks received empty pseudo_pointclouds; skipping training step.", flush=True)
                return None

            batch["pseudo_supervision_loss_weight"] = local_valid * (world_size / valid_ranks)

        res = self.step(batch, batch_idx)

        if self._compute_metrics and self.config.compute_occ3d_metrics:
            self._compute_occ3d_metrics(batch, res, "train")

        self._log_train(batch, res)

        return res

    def validation_step(self, batch: dict, batch_idx: int) -> dict:
        with torch.no_grad():
            res = self.step(batch, batch_idx)

        if self.config.compute_occ3d_metrics:
            self._compute_occ3d_metrics(batch, res, "val")

        self._log_val(batch, res)
        return res

    def test_step(self, batch: dict, batch_idx: int) -> dict:
        if self.visualization_mode:
            return {}

        with torch.no_grad():
            res = self.step(batch, batch_idx)

        if self.config.compute_occ3d_metrics:
            self._compute_occ3d_metrics(batch, res, "test")

        return res

    def optimizer_step(self, *args, **kwargs):
        try:
            super().optimizer_step(*args, **kwargs)
        except Exception as e:
            for n, p in self.net.named_parameters():
                if p.grad is None and p.requires_grad:
                    logger.warning("%s has no grad but requires grad", n)
            raise e


def detach_dict(d, skip_keys=["loss"]):
    out = {}
    for k, v in d.items():
        if k in skip_keys:
            out[k] = v
        elif isinstance(v, torch.Tensor):
            out[k] = v.detach()
        elif isinstance(v, dict):
            out[k] = detach_dict(v)  # recursive
        else:
            out[k] = v
    return out


if __name__ == "__main__":
    model = PreTrainConfig().setup()
