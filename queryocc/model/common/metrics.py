import math
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from torch.utils.cpp_extension import load
from torchmetrics import Metric

_DVR_DIR = Path(__file__).resolve().parents[1] / "ops" / "dvr"


class _LazyDvr:
    """JIT-compile the dvr extension on first use instead of at import time.

    Compilation requires a CUDA toolkit matching the torch build, which dev machines
    may not have. Importing this module must not fail there; only using the
    dvr-based metrics requires the extension.
    """

    _module = None

    def __getattr__(self, name):
        if _LazyDvr._module is None:
            _LazyDvr._module = load(
                "dvr",
                sources=[str(_DVR_DIR / "dvr.cpp"), str(_DVR_DIR / "dvr.cu")],
                verbose=True,
                extra_cflags=["-Wno-deprecated-declarations"],  # For C++ compiler
                extra_cuda_cflags=["-allow-unsupported-compiler"],
            )
        return getattr(_LazyDvr._module, name)


dvr = _LazyDvr()


class BinaryRayIoU(Metric):
    @staticmethod
    def _generate_lidar_rays():  # Keep as staticmethod
        pitch_angles = []
        for k in range(10):
            angle = math.pi / 2 - math.atan(k + 1)
            pitch_angles.append(-angle)

        while len(pitch_angles) > 0 and pitch_angles[-1] < 0.21:
            if len(pitch_angles) < 2:
                break
            delta = pitch_angles[-1] - pitch_angles[-2]
            if delta <= 0:
                break
            pitch_angles.append(pitch_angles[-1] + delta)

        lidar_rays_list = []
        for pitch_angle in pitch_angles:
            for azimuth_angle_deg in np.arange(0, 360, 1):
                azimuth_angle_rad = np.deg2rad(azimuth_angle_deg)
                x = np.cos(pitch_angle) * np.cos(azimuth_angle_rad)
                y = np.cos(pitch_angle) * np.sin(azimuth_angle_rad)
                z = np.sin(pitch_angle)
                lidar_rays_list.append((x, y, z))
        if not lidar_rays_list:
            return np.array([], dtype=np.float32).reshape(0, 3)
        return torch.tensor(lidar_rays_list, dtype=torch.float32)

    def __init__(
        self,
        depth_threshold: float,
        prob_threshold: float,
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # check parameters
        assert depth_threshold > 0.0, (
            f"The 'depth_threshold' parameter must be strictly greater than 0.0, but got {depth_threshold}."
        )
        self.register_buffer("depth_threshold", torch.tensor(depth_threshold))
        self.register_buffer("prob_threshold", torch.tensor(prob_threshold))

        self.register_buffer("lidar_rays", self._generate_lidar_rays())

        self.add_state("intersection", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(1), dist_reduce_fx="sum")

        self.register_buffer("_pc_range", torch.tensor([-40, -40, -1.0, 40, 40, 5.4]))
        self.register_buffer("_voxel_size", torch.tensor(0.4))

    def update(self, probs: torch.Tensor, labels: torch.Tensor, lidar_origins: torch.Tensor, sample_token: List[str]):
        batch_size = probs.shape[0]
        for i in range(batch_size):
            if sample_token[i] == "n/a":
                continue
            self._update_one_batch(probs[i], labels[i], lidar_origins[i].unsqueeze(0))

    def _update_one_batch(self, probs: torch.Tensor, labels: torch.Tensor, lidar_origins: torch.Tensor):
        assert labels.dtype == torch.bool, "Labels must be the binary target (bool). Got {}".format(labels.dtype)
        preds = (probs > self.prob_threshold).float()
        preds = preds.reshape([200, 200, 16])
        labels = labels.reshape([200, 200, 16])

        pcd_pred = self._process_one_sample(preds, lidar_origins)
        del preds
        pcd_gt = self._process_one_sample(labels, lidar_origins)
        del labels
        # evalute on non-free rays.
        valid_mask = pcd_gt[:, 0].int() != 0
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        self._calc_rayiou(pcd_pred, pcd_gt)

    def compute(self):
        binary_ray_iou = self.intersection / (self.union + 1e-6)  # Avoid division by zero
        return binary_ray_iou

    def _calc_rayiou(self, pcd_pred: torch.Tensor, pcd_gt: torch.Tensor):
        # L1
        depth_pred = pcd_pred[:, 1]
        depth_gt = pcd_gt[:, 1]
        l1_error = torch.abs(depth_pred - depth_gt)
        tp_dist_mask = l1_error < self.depth_threshold

        cls_id = 1
        cls_mask_pred = pcd_pred[:, 0] == cls_id
        cls_mask_gt = pcd_gt[:, 0] == cls_id

        gt_cnt = cls_mask_gt.sum()
        pred_cnt = cls_mask_pred.sum()
        tp_cls = cls_mask_gt & cls_mask_pred
        tp_mask = torch.logical_and(tp_cls, tp_dist_mask)
        tp_cnt = tp_mask.sum()

        # update states
        self.intersection += tp_cnt
        self.union += gt_cnt + pred_cnt - tp_cnt

    @torch.no_grad()
    def _process_one_sample(
        self, sem_pred: torch.Tensor, output_origin: torch.Tensor
    ) -> torch.Tensor:  # copy parameter names (that are not clear here)
        T = output_origin.shape[1]
        pred_pcds_t = []

        # binary: sem_pred contains occupancy info only
        occ_pred = sem_pred
        # dvr.render_forward() requires occ_pred to be shape of (n, ts, v.z, v.y, v.x)
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()

        offset = self._pc_range[:3].unsqueeze(0).unsqueeze(0)
        scaler = self._voxel_size.unsqueeze(0).unsqueeze(0).repeat(1, 1, 3)

        lidar_tindex = torch.zeros([1, self.lidar_rays.shape[0]], device=occ_pred.device)

        for t in range(T):
            lidar_origin = output_origin[:, t : t + 1, :]  # [1, 1, 3]
            lidar_endpts = self.lidar_rays[None] + lidar_origin  # [1, N, 3]

            output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
            output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
            output_tindex_render = lidar_tindex  # [1, N], all zeros

            with torch.no_grad():
                pred_dist, _, coord_index = dvr.render_forward(
                    occ_pred,
                    output_origin_render,
                    output_points_render,
                    output_tindex_render,
                    [1, 16, 200, 200],
                    "test",
                )
                pred_dist *= self._voxel_size

            coord_index = coord_index[0, :, :].int()  # [N, 3]

            pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
            pred_dist = pred_dist[0, :, None]
            pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)

            pred_pcds_t.append(pred_pcds)

        pred_pcds_t = torch.cat(pred_pcds_t, dim=0)  # [N*T, 2]
        return pred_pcds_t


class SemanticRayIoU(Metric):
    def __init__(
        self,
        depth_threshold: float,
        class_names: Union[tuple, list],
        dist_sync_on_step=False,
    ):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.depth_threshold = depth_threshold

        self.register_buffer("lidar_rays", BinaryRayIoU._generate_lidar_rays())
        self.register_buffer("_pc_range", torch.tensor([-40, -40, -1.0, 40, 40, 5.4]))
        self.register_buffer("_voxel_size", torch.tensor(0.4))
        self.class_names = ["noise"] + list(class_names) + ["free"]

        self.add_state("intersection", default=torch.zeros(len(self.class_names)), dist_reduce_fx="sum")
        self.add_state("union", default=torch.zeros(len(self.class_names)), dist_reduce_fx="sum")

    @torch.no_grad()
    def update(
        self, sem_preds: torch.Tensor, sem_gts: torch.Tensor, lidar_origins: torch.Tensor, sample_token: List[str]
    ):
        batch_size = sem_preds.shape[0]
        for i in range(batch_size):
            if sample_token[i] == "n/a":
                continue
            self._update_one_batch(sem_preds[i], sem_gts[i], lidar_origins[i].unsqueeze(0))

    def _update_one_batch(self, sem_pred: torch.Tensor, sem_gt: torch.Tensor, lidar_origins: torch.Tensor):
        sem_pred = sem_pred.reshape([200, 200, 16])
        sem_gt = sem_gt.reshape([200, 200, 16])

        pcd_pred = self._process_one_sample(sem_pred, lidar_origins)
        del sem_pred
        pcd_gt = self._process_one_sample(sem_gt, lidar_origins)
        del sem_gt, lidar_origins

        # evalute on non-free rays
        valid_mask = pcd_gt[:, 0].int() != len(self.class_names) - 1
        pcd_pred = pcd_pred[valid_mask]
        pcd_gt = pcd_gt[valid_mask]

        assert pcd_pred.shape == pcd_gt.shape
        self._calc_rayiou(pcd_pred, pcd_gt)

    def _process_one_sample(self, sem_pred: torch.Tensor, output_origin: torch.Tensor) -> torch.Tensor:
        T = output_origin.shape[1]
        pred_pcds_t = []
        free_id = len(self.class_names) - 1  # free_id = 15 for DEFAULT_NUSCENES_CLASSES
        occ_pred = torch.zeros_like(sem_pred)
        occ_pred[sem_pred < free_id] = 1
        occ_pred[sem_pred == free_id] = 0
        occ_pred = occ_pred.permute(2, 1, 0)
        occ_pred = occ_pred[None, None, :].contiguous().float()

        offset = self._pc_range[:3].unsqueeze(0).unsqueeze(0)
        scaler = self._voxel_size.unsqueeze(0).unsqueeze(0).repeat(1, 1, 3)

        lidar_tindex = torch.zeros([1, self.lidar_rays.shape[0]], device=occ_pred.device)

        for t in range(T):
            lidar_origin = output_origin[:, t : t + 1, :]  # [1, 1, 3]
            lidar_endpts = self.lidar_rays[None] + lidar_origin  # [1, N, 3]

            output_origin_render = ((lidar_origin - offset) / scaler).float()  # [1, 1, 3]
            output_points_render = ((lidar_endpts - offset) / scaler).float()  # [1, N, 3]
            output_tindex_render = lidar_tindex  # [1, N], all zeros

            with torch.no_grad():
                pred_dist, _, coord_index = dvr.render_forward(
                    occ_pred,
                    output_origin_render,
                    output_points_render,
                    output_tindex_render,
                    [1, 16, 200, 200],
                    "test",
                )
                pred_dist *= self._voxel_size

            coord_index = coord_index[0, :, :].int()  # [N, 3]

            pred_label = sem_pred[coord_index[:, 0], coord_index[:, 1], coord_index[:, 2]][:, None]  # [N, 1]
            pred_dist = pred_dist[0, :, None]
            pred_pcds = torch.cat([pred_label.float(), pred_dist], dim=-1)

            pred_pcds_t.append(pred_pcds)

        pred_pcds_t = torch.cat(pred_pcds_t, dim=0)  # [N*T, 2]
        return pred_pcds_t

    def _calc_rayiou(self, pcd_pred: torch.Tensor, pcd_gt: torch.Tensor):
        # L1
        depth_pred = pcd_pred[:, 1]
        depth_gt = pcd_gt[:, 1]
        l1_error = torch.abs(depth_pred - depth_gt)
        del depth_pred, depth_gt
        tp_dist_mask = l1_error < self.depth_threshold

        num_classes = len(self.class_names)
        pred_cls_ids = pcd_pred[:, 0].long()  # [N_rays]
        gt_cls_ids = pcd_gt[:, 0].long()  # [N_rays]

        gt_cnt_vec = torch.bincount(gt_cls_ids, minlength=num_classes)  # [num_classes]
        pred_cnt_vec = torch.bincount(pred_cls_ids, minlength=num_classes)  # [num_classes]

        tp_cls_match = gt_cls_ids.clone()
        mismatched_mask = pred_cls_ids != gt_cls_ids
        tp_cls_match[mismatched_mask] = num_classes  # Assign an index outside the valid range
        tp_weights = tp_dist_mask.float()
        tp_cnt_vec = torch.bincount(tp_cls_match, weights=tp_weights, minlength=num_classes + 1)
        tp_cnt_vec = tp_cnt_vec[:num_classes]
        self.intersection.add_(tp_cnt_vec)
        self.union.add_(gt_cnt_vec + pred_cnt_vec - tp_cnt_vec)

    def compute(self):
        iou_list = self.intersection / (self.union + 1e-6)  # Uses accumulated totals
        return iou_list


# Example Usage
if __name__ == "__main__":
    ray_iou = BinaryRayIoU(depth_threshold=1, prob_threshold=0.25)
    print("generate lidar rays:", ray_iou.lidar_rays.shape)
