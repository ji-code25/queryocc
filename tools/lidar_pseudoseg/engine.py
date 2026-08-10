"""Offline generation of per-point lidar semantic pseudo-labels.

Camera semantic masks (produced by ``generate_pseudo_labels.generate_semantics``) are
projected onto the lidar point cloud of the same sample, giving one nuScenes-lidarseg
class id per lidar point. The result is written as ``{token}_pseudoseg.bin``
(``uint8``, one label per point, in point order) and is consumed at training time by
the lidar-supervised arms (queryocc-lidar-nusc, queryocc-joint-nusc, queryocc-nusc-av2)
via ``paths.NUSC_PSEUDOSEG_ROOT`` / ``paths.AV2_PSEUDOSEG_ROOT``.

Run with::

    python -m tools.lidar_pseudoseg.engine --dataset nuscenes ...

See ``tools/lidar_pseudoseg/README.md``.
"""

import argparse
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

if __package__ in (None, ""):  # allow `python tools/lidar_pseudoseg/engine.py` as well as `-m`
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from queryocc.common import paths
from queryocc.data.datasets.nuscenes_mappings import NUSCENES_IDX_TO_LABELS_MAP, NUSCENES_LABELS_TO_IDX_MAP, LabelMapper
from queryocc.data.vfm.grounding_sam_reader import GroundingSAMReader, GroundingSAMReaderConfig
from tools.lidar_pseudoseg.utils import (
    _inverse,
    _transform,
    batched_depth_filtering,
    batched_project_kannala,
    batched_project_pinhole,
    get_argoverse_pc,
    get_camera_extrinsics,
    get_camera_intrinsics,
    get_ego_pose,
    get_lidar_extrinsics,
    get_lidar_pose,
    get_nuscenes_pc,
    voxel_based_relabeling,
)


def _av2_lidar_token(sample: Dict) -> str:
    """Stem of the AV2 pseudoseg filename: the lidar timestamp, which create_infos_av2 stores as `token`."""
    return sample.get("lidar_token") or sample["token"]


@dataclass
class PseudoLabelerConfig:
    pseudoseg_root: Path = paths.NUSC_PSEUDOSEG_ROOT
    """Directory holding the per-run pseudoseg directories (<pseudoseg_root>/<pseudo_label_name>[_fused])."""

    dataset: Literal["nuscenes", "argoverse2"] = "nuscenes"
    """The dataset to use."""

    split: Literal["train", "val"] = "train"
    """The split to use."""

    prefix: str = ""
    """The prefix for the nuscenes version, e.g. '-mini' for v1.0-mini."""

    infos_pkl: Optional[Path] = None
    """Explicit infos pkl (from generate_pseudo_labels.create_infos). Defaults to the standard name."""

    grounding_sam_reader: GroundingSAMReaderConfig = field(default_factory=GroundingSAMReaderConfig)
    """Reader for the semantics npz stream (<semantics_root>/<scene_name>/<token>.npz)."""

    extra_semantics_roots: Tuple[Path, ...] = ()
    """Additional semantics roots, searched in order when a camera is missing from the primary one.

    Needed for Argoverse 2: its ring cameras do not share a resolution, so the semantics are generated
    one resolution group at a time into separate roots (see generate_pseudo_labels/README.md)."""

    nuscenes_data: Path = paths.NUSC_ROOT
    """The path to the nuscenes data."""

    argoverse_data: Path = paths.AV2_ROOT
    """The path to the argoverse data."""

    fusion_method: Literal["last_cam", "max"] = "max"
    """The fusion method to use."""

    scenewise_fusion: bool = True
    """Whether to fuse the pseudosegs scene-wise."""

    voxel_size: float = 1.0
    """The voxel size to use for the voxel based relabeling (m)"""

    pseudo_label_name: str = "delta"
    """The name of the pseudo label to save."""

    override: bool = False
    """Whether to override existing pseudosegs."""

    depth_scaling: float = 1.2
    """How far away (factor) a point can be to still be considered ok in background removal."""

    image_downscaling: int = 16
    """How much to downsample the image before checking which pixels have been used. Larger factor increases the region in which project points are rejected"""

    with_score: bool = False
    """Whether the grounding-sam masks have corresponding confidence scores."""

    num_workers: int = 16
    """Number of worker threads used when creating per-sample pseudosegs."""


class PseudoLabelerEngine:
    def __init__(self, config: PseudoLabelerConfig):
        self.config = config

        self.grounding_sam_reader = GroundingSAMReader(config.grounding_sam_reader, self.config.dataset)
        self.grounding_sam_readers = [self.grounding_sam_reader] + [
            GroundingSAMReader(replace(config.grounding_sam_reader, semantics_root=Path(root)), self.config.dataset)
            for root in config.extra_semantics_roots
        ]

        self.label_mapper = LabelMapper(list(NUSCENES_IDX_TO_LABELS_MAP.values())[1:-2])

    # ------------------------------------------------------------------ paths

    @property
    def pseudoseg_dir(self) -> Path:
        return self.config.pseudoseg_root / self.config.pseudo_label_name

    @property
    def fused_pseudoseg_dir(self) -> Path:
        return self.config.pseudoseg_root / f"{self.config.pseudo_label_name}_fused"

    # ------------------------------------------------------------------ masks

    def _first_available(self, scene_name: str, token: str, camera_name: str, scores: bool) -> Optional[np.ndarray]:
        """The camera's array from the first semantics root that has it (see extra_semantics_roots)."""
        for reader in self.grounding_sam_readers:
            array = (
                reader.read_camera_scores(scene_name, token, camera_name)
                if scores
                else reader.read_camera(scene_name, token, camera_name)
            )
            if array is not None:
                return array
        return None

    def _read_camera_masks(
        self, scene_name: str, token: str, camera_names: List[str]
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Semantic masks per camera, or None when this sample has no generated labels."""
        masks = {}
        for camera_name in camera_names:
            mask = self._first_available(scene_name, token, camera_name, scores=False)
            if mask is None:
                return None
            # nuScenes-lidarseg class ids (0..18) fit in int8, which is what the projection expects.
            masks[camera_name] = torch.as_tensor(np.ascontiguousarray(mask, dtype=np.int8))
        return masks

    def _read_camera_scores(
        self, scene_name: str, token: str, camera_names: List[str]
    ) -> Optional[Dict[str, torch.Tensor]]:
        scores = {}
        for camera_name in camera_names:
            score = self._first_available(scene_name, token, camera_name, scores=True)
            if score is None:
                return None
            scores[camera_name] = torch.as_tensor(np.ascontiguousarray(score, dtype=np.float16))
        return scores

    # ------------------------------------------------------------------ fusion

    def _fuse_pseudosegs(
        self, classes_per_cam: Dict[str, torch.Tensor], active_per_cam: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        pseudoseg = torch.ones(classes_per_cam[list(classes_per_cam.keys())[0]].shape[0], dtype=torch.int8) * -1
        if self.config.fusion_method == "last_cam":
            for cam_name, classes in classes_per_cam.items():
                active = active_per_cam[cam_name]
                pseudoseg[active] = classes[active]
        elif self.config.fusion_method == "max":
            for cam_name, classes in classes_per_cam.items():
                current_active = active_per_cam[cam_name]
                current_classes = classes[current_active]
                existing_classes = pseudoseg[current_active]
                max_classes = torch.maximum(current_classes, existing_classes)  # This selection logic is not reasonable
                pseudoseg[current_active] = max_classes

        return pseudoseg

    def _fuse_pseudosegs_with_scores(
        self,
        classes_per_cam: Dict[str, torch.Tensor],
        scores_per_cam: Dict[str, torch.Tensor],
        active_per_cam: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        pseudoseg = torch.ones(classes_per_cam[list(classes_per_cam.keys())[0]].shape[0], dtype=torch.int8) * -1
        pseudoseg_scores = torch.ones_like(pseudoseg, dtype=torch.float16) * -1
        for cam_name, classes in classes_per_cam.items():
            current_active = active_per_cam[cam_name]
            current_classes = classes[current_active]
            existing_classes = pseudoseg[current_active]
            current_scores = scores_per_cam[cam_name][current_active]
            existing_scores = pseudoseg_scores[current_active]

            # Find where current scores are larger
            higher_score_mask = current_scores > existing_scores
            # Select classes based on which score is higher
            selected_classes = torch.where(higher_score_mask, current_classes, existing_classes)
            pseudoseg[current_active] = selected_classes
            # Update scores with the larger values
            selected_scores = torch.where(higher_score_mask, current_scores, existing_scores)
            pseudoseg_scores[current_active] = selected_scores
        return pseudoseg, pseudoseg_scores

    # ------------------------------------------------------------------ nuScenes

    def _get_nuscenes_data(self, split: Literal["train", "val"]) -> List[Dict]:
        infos_pkl = self.config.infos_pkl
        if infos_pkl is None:
            infos_pkl = paths.INFOS_DIR / f"bevdetv2-nuscenes{self.config.prefix}_infos_{split}.pkl"
        print(f"Loading nuScenes infos from {infos_pkl}")
        with open(infos_pkl, "rb") as f:
            nusc_data = pickle.load(f)["infos"]
        return nusc_data

    def _run_nuscenes(self):
        nusc_data = self._get_nuscenes_data(split=self.config.split)
        # Check if pseudoseg folder already exists
        if self.pseudoseg_dir.exists() and not self.config.override:
            print(f"Pseudoseg folder {self.pseudoseg_dir} already exists. Skipping creation step.")
        else:
            print(f"Creating pseudosegs in {self.pseudoseg_dir}")
            self._create_nuscenes_pseudosegs(nusc_data)

        if self.config.scenewise_fusion:
            self._fuse_nuscenes_sequences_pseudosegs(nusc_data)

    def _create_nuscenes_pseudosegs(self, nusc_data: List[Dict]):
        num_ok = 0
        num_skipped = 0

        # Multi-threaded
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(self._create_nuscenes_pseudoseg_from_sample, sample) for sample in nusc_data]
            for future in tqdm(as_completed(futures), total=len(futures)):
                num_ok += future.result()[0]
                num_skipped += future.result()[1]
        print(f"Created pseudosegs: {num_ok} ok, {num_skipped} skipped")
        return num_ok, num_skipped

    def _create_nuscenes_pseudoseg_from_sample(self, sample: Dict):
        scene_name = sample["scene_name"]
        token = sample["token"]
        lidar_token = sample["lidar_idx"]

        pseudoseg_fpath = self.pseudoseg_dir / f"{lidar_token}_pseudoseg.bin"
        pseudoseg_score_fpath = self.pseudoseg_dir / f"{lidar_token}_pseudoseg_scores.bin"
        pseudoseg_fpath.parent.mkdir(parents=True, exist_ok=True)

        if pseudoseg_fpath.exists() and not self.config.override:
            num_ok = 1
            num_skipped = 0
            return num_ok, num_skipped

        camera_names = list(sample["cams"].keys())
        image_masks = self._read_camera_masks(scene_name, token, camera_names)
        image_scores = self._read_camera_scores(scene_name, token, camera_names) if self.config.with_score else None
        if image_masks is None or (self.config.with_score and image_scores is None):
            print(f"Image masks not found for scene {scene_name}, token {token}. Skipping sample.")
            num_ok = 0
            num_skipped = 1
            return num_ok, num_skipped
        num_ok = 1
        num_skipped = 0

        points_lidar, relevant_points_mask = get_nuscenes_pc(sample, self.config.nuscenes_data)
        num_points = points_lidar.shape[0]

        T_ego_lidar = get_lidar_extrinsics(sample)

        camera_extrinsics = get_camera_extrinsics(sample)
        camera_intrinsics = get_camera_intrinsics(sample)

        T_ego_lidar = T_ego_lidar.unsqueeze(0).unsqueeze(0).repeat(1, num_points, 1, 1)
        points_ego = _transform(points_lidar.clone().unsqueeze(0), T_ego_lidar)

        classes_per_cam = {}
        active_per_cam = {}
        scores_per_cam = {}
        for cam_name in camera_extrinsics.keys():
            T_cam_ego = camera_extrinsics[cam_name]
            T_ego_cam = _inverse(T_cam_ego)
            T_ego_cam = T_ego_cam.unsqueeze(0).unsqueeze(0).repeat(1, num_points, 1, 1)
            points_cam = _transform(points_ego.clone(), T_ego_cam)

            points_to_project_mask = relevant_points_mask.unsqueeze(0) & (points_cam[..., 2] > 0)
            point_cam_to_project = points_cam[points_to_project_mask, :].unsqueeze(0)
            num_points_to_project = point_cam_to_project.shape[1]

            T_img_cam = camera_intrinsics[cam_name].unsqueeze(0)
            xy_array = batched_project_kannala(point_cam_to_project[..., :3], T_img_cam)
            xyd_array = torch.cat([xy_array, point_cam_to_project[..., 2:3]], dim=-1)

            image_mask = image_masks[cam_name]
            image_size_h = image_mask.shape[0]
            image_size_w = image_mask.shape[1]
            inside_mask = torch.logical_and(
                torch.logical_and(xyd_array[..., 0] >= 0, xyd_array[..., 0] < image_size_w),
                torch.logical_and(xyd_array[..., 1] >= 0, xyd_array[..., 1] < image_size_h),
            )

            dummy_point = torch.tensor([0, 0, 1e6], dtype=xyd_array.dtype, device=points_cam.device)
            xyd_array[~inside_mask] = dummy_point
            foreground_masks = batched_depth_filtering(
                xyd_array, (image_size_w, image_size_h), self.config.depth_scaling, self.config.image_downscaling
            )
            inside_mask = torch.logical_and(inside_mask, foreground_masks)

            inside_xy_points = xy_array[inside_mask].to(torch.int32)
            classes = image_mask[inside_xy_points[..., 1], inside_xy_points[..., 0]]

            all_active = torch.zeros_like(relevant_points_mask)
            all_active[points_to_project_mask.squeeze(0)] = inside_mask.squeeze(0)

            all_classes_projected = torch.ones(num_points_to_project, dtype=torch.int8) * -1
            all_classes_projected[inside_mask.squeeze(0)] = classes
            all_classes = torch.ones(num_points, dtype=torch.int8) * -1
            all_classes[points_to_project_mask.squeeze(0)] = all_classes_projected

            classes_per_cam[cam_name] = all_classes
            active_per_cam[cam_name] = all_active

            if self.config.with_score:
                image_score = image_scores[cam_name]
                scores = image_score[inside_xy_points[..., 1], inside_xy_points[..., 0]]
                all_scores_projected = torch.zeros(num_points_to_project, dtype=torch.float16)
                all_scores_projected[inside_mask.squeeze(0)] = scores
                all_scores = torch.zeros(num_points, dtype=torch.float16)
                all_scores[points_to_project_mask.squeeze(0)] = all_scores_projected
                scores_per_cam[cam_name] = all_scores

        if self.config.with_score:
            pseudoseg, pseudoseg_scores = self._fuse_pseudosegs_with_scores(
                classes_per_cam, scores_per_cam, active_per_cam
            )
            pseudoseg_scores[pseudoseg_scores == -1] = 0
            pseudoseg_scores = pseudoseg_scores.cpu().numpy().astype(np.float16).reshape(-1, 1)
            pseudoseg_scores.tofile(pseudoseg_score_fpath)
        else:
            pseudoseg = self._fuse_pseudosegs(classes_per_cam, active_per_cam)

        pseudoseg[pseudoseg == -1] = NUSCENES_LABELS_TO_IDX_MAP["free"]
        pseudoseg = pseudoseg.cpu().numpy().astype(np.uint8).reshape(-1, 1)
        pseudoseg.tofile(pseudoseg_fpath)

        return num_ok, num_skipped

    def _fuse_nuscenes_sequences_pseudosegs(self, nusc_data: List[Dict]):
        fused_dir = self.fused_pseudoseg_dir
        fused_dir.mkdir(parents=True, exist_ok=True)

        scene_names = list(set([sample["scene_name"] for sample in nusc_data]))
        nusc_data_by_scene = {
            scene_name: [sample for sample in nusc_data if sample["scene_name"] == scene_name]
            for scene_name in scene_names
        }

        for scene_name, scene_data in tqdm(
            nusc_data_by_scene.items(), total=len(nusc_data_by_scene), desc="Fusing sequence wise pseudosegs"
        ):
            # Check if all fused pseudoseg fpaths already exists
            fused_pseudoseg_fpaths = [fused_dir / f"{sample['lidar_idx']}_pseudoseg.bin" for sample in scene_data]
            if (
                all(fused_pseudoseg_fpath.exists() for fused_pseudoseg_fpath in fused_pseudoseg_fpaths)
                and not self.config.override
            ):
                continue

            pseudoseg_fpaths = [self.pseudoseg_dir / f"{sample['lidar_idx']}_pseudoseg.bin" for sample in scene_data]
            if not all(pseudoseg_fpath.exists() for pseudoseg_fpath in pseudoseg_fpaths):
                print(f"Missing per-sample pseudosegs for scene {scene_name}. Skipping scene.")
                continue

            pcds_and_masks = [get_nuscenes_pc(sample, self.config.nuscenes_data) for sample in scene_data]
            pcds = [pcd for pcd, _ in pcds_and_masks]
            num_points = [pcd.shape[0] for pcd in pcds]

            lidar_poses = [get_lidar_pose(sample) for sample in scene_data]

            pseudolabels = [
                torch.tensor(np.fromfile(pseudoseg_fpath, dtype=np.uint8).reshape(-1, 1), dtype=torch.long)
                for pseudoseg_fpath in pseudoseg_fpaths
            ]

            # Select middle frame as main frame
            main_frame_idx = len(scene_data) // 2
            main_pose = lidar_poses[main_frame_idx]

            # Get relative poses
            relative_poses = [_inverse(main_pose) @ lidar_pose for lidar_pose in lidar_poses]

            # Transform pcds to main frame
            pcds = [_transform(pcd, relative_pose) for pcd, relative_pose in zip(pcds, relative_poses)]
            pcds = torch.cat(pcds, dim=0)
            pseudolabels = torch.cat(pseudolabels, dim=0)
            fused_pseudolabels = pseudolabels.clone()

            # Get static points mask
            static_points_mask = self.label_mapper.nusc_to_static(pseudolabels).squeeze()
            dynamic_points_mask = self.label_mapper.nusc_to_dynamic(pseudolabels).squeeze()
            noise_or_free_points_mask = static_points_mask & dynamic_points_mask

            # Vote for static vs dynamic
            static_votes = static_points_mask & ~noise_or_free_points_mask

            static_results = voxel_based_relabeling(pcds, static_votes.long(), 2, self.config.voxel_size)
            dynamic_results = static_results == 0

            # Override the dynamic points that were incorrectly labeled as dynamic as noise
            voted_dynamic_points_mask = dynamic_points_mask & ~dynamic_results & ~noise_or_free_points_mask
            fused_pseudolabels[voted_dynamic_points_mask] = self.label_mapper._nusc_label_to_idx["noise"]

            # Vote for static environment classes
            fused_static_points_mask = self.label_mapper.nusc_to_static(fused_pseudolabels).squeeze()

            fused_static_points = pcds[fused_static_points_mask, :]
            fused_static_pseudolabels = fused_pseudolabels[fused_static_points_mask]
            fused_static_results = voxel_based_relabeling(
                fused_static_points,
                fused_static_pseudolabels.squeeze().long(),
                len(self.label_mapper._nusc_idx_to_label),
                self.config.voxel_size,
                skip_idx=self.label_mapper._nusc_label_to_idx["free"],
            )
            fused_pseudolabels[fused_static_points_mask, 0] = fused_static_results

            # Put back the points in the list per sample
            accumulated_points = 0
            for sample, nps in zip(scene_data, num_points):
                fused_pseudoseg_fpath = fused_dir / f"{sample['lidar_idx']}_pseudoseg.bin"
                fused_pseudoseg = fused_pseudolabels[accumulated_points : accumulated_points + nps]
                fused_pseudoseg = fused_pseudoseg.cpu().numpy().astype(np.uint8).reshape(-1, 1)
                fused_pseudoseg.tofile(fused_pseudoseg_fpath)
                accumulated_points += nps

    # ------------------------------------------------------------------ Argoverse 2

    def _get_argoverse_data(self, split: Literal["train", "val"]) -> List[Dict]:
        infos_pkl = self.config.infos_pkl
        if infos_pkl is None:
            infos_pkl = self.config.argoverse_data / f"av2_infos_{split}.pkl"
        print(f"Loading Argoverse 2 infos from {infos_pkl}")
        with open(infos_pkl, "rb") as f:
            av2_data = pickle.load(f)
        # generate_pseudo_labels.create_infos_av2 writes {"infos": [...], "metadata": {...}}; a
        # bare list (an older index format) is also tolerated.
        return av2_data["infos"] if isinstance(av2_data, dict) else av2_data

    def _run_argoverse2(self):
        av2_data = self._get_argoverse_data(split=self.config.split)

        # Check if pseudoseg folder already exists
        if self.pseudoseg_dir.exists() and not self.config.override:
            print(f"Pseudoseg folder {self.pseudoseg_dir} already exists. Skipping creation step.")
        else:
            print(f"Creating pseudosegs in {self.pseudoseg_dir}")
            self._create_argoverse_pseudosegs(av2_data)

        if self.config.scenewise_fusion:
            self._fuse_argoverse_sequences_pseudosegs(av2_data)

    def _fuse_argoverse_sequences_pseudosegs(self, av2_data: List[Dict]):
        fused_dir = self.fused_pseudoseg_dir
        fused_dir.mkdir(parents=True, exist_ok=True)

        scene_names = list(set([sample["scene_name"] for sample in av2_data]))
        av2_data_by_scene = {
            scene_name: [sample for sample in av2_data if sample["scene_name"] == scene_name]
            for scene_name in scene_names
        }

        for scene_name, scene_data in tqdm(
            av2_data_by_scene.items(), total=len(av2_data_by_scene), desc="Fusing sequence wise pseudosegs"
        ):
            # Check if all fused pseudoseg fpaths already exists
            fused_pseudoseg_fpaths = [
                fused_dir / scene_name / f"{_av2_lidar_token(sample)}_pseudoseg.bin" for sample in scene_data
            ]
            if (
                all(fused_pseudoseg_fpath.exists() for fused_pseudoseg_fpath in fused_pseudoseg_fpaths)
                and not self.config.override
            ):
                continue

            # Load pcds and masks for all samples in the scene
            pseudoseg_fpaths = [
                self.pseudoseg_dir / scene_name / f"{_av2_lidar_token(sample)}_pseudoseg.bin" for sample in scene_data
            ]
            if not all(pseudoseg_fpath.exists() for pseudoseg_fpath in pseudoseg_fpaths):
                print(f"Missing per-sample pseudosegs for scene {scene_name}. Skipping scene.")
                continue

            pcds_and_masks = [get_argoverse_pc(sample, self.config.argoverse_data) for sample in scene_data]
            pcds = [pcd for pcd, _ in pcds_and_masks]
            num_points = [pcd.shape[0] for pcd in pcds]

            # Get ego poses for all samples in the scene (since AV2 points are in ego frame)
            ego_poses = [get_ego_pose(sample) for sample in scene_data]

            # Load pseudolabels for all samples in the scene
            pseudolabels = [
                torch.tensor(np.fromfile(pseudoseg_fpath, dtype=np.uint8).reshape(-1, 1), dtype=torch.long)
                for pseudoseg_fpath in pseudoseg_fpaths
            ]

            # Select middle frame as main frame
            main_frame_idx = len(scene_data) // 2
            main_pose = ego_poses[main_frame_idx]

            # Get relative poses
            relative_poses = [_inverse(main_pose) @ ego_pose for ego_pose in ego_poses]

            # Transform pcds to main frame
            pcds = [_transform(pcd, relative_pose) for pcd, relative_pose in zip(pcds, relative_poses)]
            pcds = torch.cat(pcds, dim=0)
            pseudolabels = torch.cat(pseudolabels, dim=0)
            fused_pseudolabels = pseudolabels.clone()

            # Get static points mask
            static_points_mask = self.label_mapper.nusc_to_static(
                pseudolabels
            ).squeeze()  # True for [1, 8, 11, 12, 13, 14, 15, 16, 17, 18]
            dynamic_points_mask = self.label_mapper.nusc_to_dynamic(
                pseudolabels
            ).squeeze()  # True for [2, 3, 4, 5, 6, 7, 9, 10, 17]
            free_points_mask = static_points_mask & dynamic_points_mask  # label 17

            # Vote for static vs dynamic
            static_votes = static_points_mask & ~free_points_mask
            static_results = voxel_based_relabeling(pcds, static_votes.long(), 2, self.config.voxel_size)

            # Override the dynamic points as noise that were incorrectly labeled as dynamic
            incorrect_dynamic_points_mask = dynamic_points_mask & ~free_points_mask & static_results
            fused_pseudolabels[incorrect_dynamic_points_mask] = self.label_mapper._nusc_label_to_idx["noise"]

            # Vote for static environment classes
            fused_static_points_mask = self.label_mapper.nusc_to_static(fused_pseudolabels).squeeze()
            fused_static_points_mask = fused_static_points_mask & ~free_points_mask  # exclude free points

            fused_static_points = pcds[fused_static_points_mask, :]
            fused_static_pseudolabels = fused_pseudolabels[fused_static_points_mask]
            fused_static_results = voxel_based_relabeling(
                fused_static_points,
                fused_static_pseudolabels.squeeze().long(),
                len(self.label_mapper._nusc_idx_to_label),
                self.config.voxel_size,
                # NB: unlike the nuScenes pass, skip_idx is not set here, so "free" votes
                # participate in the relabeling and free-labeled points end up as noise(0).
            )
            fused_pseudolabels[fused_static_points_mask, 0] = fused_static_results

            # Put back the points in the list per sample
            accumulated_points = 0
            for sample, nps in zip(scene_data, num_points):
                fused_pseudoseg_fpath = fused_dir / scene_name / f"{_av2_lidar_token(sample)}_pseudoseg.bin"
                fused_pseudoseg_fpath.parent.mkdir(parents=True, exist_ok=True)
                fused_pseudoseg = fused_pseudolabels[accumulated_points : accumulated_points + nps]
                fused_pseudoseg = fused_pseudoseg.cpu().numpy().astype(np.uint8).reshape(-1, 1)
                fused_pseudoseg.tofile(fused_pseudoseg_fpath)
                accumulated_points += nps

    def _create_argoverse_pseudosegs(self, av2_data: List[Dict]):
        num_ok = 0
        num_skipped = 0

        # Multi-threaded
        with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
            futures = [executor.submit(self._create_argoverse_pseudoseg_from_sample, sample) for sample in av2_data]
            for future in tqdm(as_completed(futures), total=len(futures)):
                num_ok += future.result()[0]
                num_skipped += future.result()[1]

        print(f"Created pseudosegs: {num_ok} ok, {num_skipped} skipped")
        return num_ok, num_skipped

    def _create_argoverse_pseudoseg_from_sample(self, sample: Dict):
        scene_name = sample["scene_name"]
        # The semantics npz is keyed by the sample token, which for AV2 is the lidar timestamp.
        token = sample["token"]
        lidar_token = _av2_lidar_token(sample)

        pseudoseg_fpath = self.pseudoseg_dir / scene_name / f"{lidar_token}_pseudoseg.bin"
        pseudoseg_fpath.parent.mkdir(parents=True, exist_ok=True)

        if pseudoseg_fpath.exists() and not self.config.override:
            num_ok = 1
            num_skipped = 0
            return num_ok, num_skipped

        camera_names = list(sample["cams"].keys())
        image_masks = self._read_camera_masks(scene_name, token, camera_names)
        if image_masks is None:
            num_ok = 0
            num_skipped = 1
            return num_ok, num_skipped
        num_ok = 1
        num_skipped = 0

        # 0. Load point cloud and calibration data
        points_lidar, relevant_points_mask = get_argoverse_pc(sample, self.config.argoverse_data)
        num_points = points_lidar.shape[0]
        camera_extrinsics = get_camera_extrinsics(sample)
        camera_intrinsics = get_camera_intrinsics(sample)

        # 1. In argoverse2, the lidar points are already in ego frame (super important!), so
        #    lidar2ego_* from the infos pkl must NOT be applied here.
        points_ego = points_lidar.unsqueeze(0)

        classes_per_cam = {}
        active_per_cam = {}
        for cam_name in camera_extrinsics.keys():
            # 2. transform points to camera frame (the infos store sensor->ego, so invert)
            T_cam_ego = camera_extrinsics[cam_name]
            T_ego_cam = _inverse(T_cam_ego)
            T_ego_cam = T_ego_cam.unsqueeze(0).unsqueeze(0).repeat(1, num_points, 1, 1)
            points_cam = _transform(points_ego.clone(), T_ego_cam)

            # 3. project points in camera coordinate to image plane
            points_to_project_mask = relevant_points_mask.unsqueeze(0) & (points_cam[..., 2] > 0)
            point_cam_to_project = points_cam[points_to_project_mask, :].unsqueeze(0)
            num_points_to_project = point_cam_to_project.shape[1]

            T_img_cam = camera_intrinsics[cam_name].unsqueeze(0)
            xy_array = batched_project_pinhole(point_cam_to_project[..., :3], T_img_cam)
            xyd_array = torch.cat([xy_array, point_cam_to_project[..., 2:3]], dim=-1)

            # a boolean mask for points inside image dimensions
            image_mask = image_masks[cam_name]
            image_size_h = image_mask.shape[0]
            image_size_w = image_mask.shape[1]
            inside_mask = torch.logical_and(
                torch.logical_and(xyd_array[..., 0] >= 0, xyd_array[..., 0] < image_size_w),
                torch.logical_and(xyd_array[..., 1] >= 0, xyd_array[..., 1] < image_size_h),
            )

            # depth filtering: mask out points that are too far
            dummy_point = torch.tensor(
                [0, 0, 1e6], dtype=xyd_array.dtype, device=points_cam.device
            )  # a dummy far away point
            xyd_array[~inside_mask] = dummy_point
            foreground_masks = batched_depth_filtering(
                xyd_array, (image_size_w, image_size_h), self.config.depth_scaling, self.config.image_downscaling
            )
            inside_mask = torch.logical_and(inside_mask, foreground_masks)

            # semantic labeling
            inside_xy_points = xy_array[inside_mask].to(torch.int32)
            classes = image_mask[inside_xy_points[..., 1], inside_xy_points[..., 0]]

            all_active = torch.zeros_like(relevant_points_mask)
            all_active[points_to_project_mask.squeeze(0)] = inside_mask.squeeze(0)

            all_classes_projected = torch.ones(num_points_to_project, dtype=torch.int8) * -1
            all_classes_projected[inside_mask.squeeze(0)] = classes
            all_classes = torch.ones(num_points, dtype=torch.int8) * -1
            all_classes[points_to_project_mask.squeeze(0)] = all_classes_projected

            classes_per_cam[cam_name] = all_classes
            active_per_cam[cam_name] = all_active

        pseudoseg = self._fuse_pseudosegs(classes_per_cam, active_per_cam)
        pseudoseg[pseudoseg == -1] = NUSCENES_LABELS_TO_IDX_MAP["free"]
        pseudoseg = pseudoseg.cpu().numpy().astype(np.uint8).reshape(-1, 1)
        pseudoseg.tofile(pseudoseg_fpath)

        return num_ok, num_skipped

    def run(self):
        if self.config.dataset == "nuscenes":
            self._run_nuscenes()
        elif self.config.dataset == "argoverse2":
            self._run_argoverse2()
        else:
            raise ValueError(f"Invalid dataset: {self.config.dataset}")


def parse_args():
    parser = argparse.ArgumentParser(description="Lidar pseudo-segmentation generator")

    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="Directory that holds the <pseudo_label_name>[_fused] pseudoseg directories. "
        "Defaults to paths.NUSC_PSEUDOSEG_ROOT (nuscenes) / paths.AV2_PSEUDOSEG_ROOT.parent (argoverse2).",
    )

    parser.add_argument(
        "--dataset", type=str, choices=["nuscenes", "argoverse2"], default="nuscenes", help="The dataset to use"
    )

    parser.add_argument("--version", default="v1.0-trainval", type=str)

    parser.add_argument("--split", type=str, choices=["train", "val"], default="train", help="The split to use")

    parser.add_argument(
        "--nuscenes_data",
        type=str,
        default=None,
        help="The path to the nuscenes data directory (default: paths.NUSC_ROOT)",
    )

    parser.add_argument(
        "--argoverse_data",
        type=str,
        default=None,
        help="The path to the argoverse data directory, used to resolve relative paths from the infos pkl and as "
        "the default location of av2_infos_<split>.pkl (default: paths.AV2_ROOT)",
    )

    parser.add_argument(
        "--infos_pkl",
        type=str,
        default=None,
        help="Infos pkl produced by generate_pseudo_labels.create_infos (nuscenes; default: standard name under "
        "paths.INFOS_DIR) or generate_pseudo_labels.create_infos_av2 (argoverse2; default: "
        "<argoverse_data>/av2_infos_<split>.pkl)",
    )

    parser.add_argument(
        "--semantics_root",
        type=str,
        nargs="+",
        default=None,
        help="Root(s) of the semantics npz stream, <root>/<scene_name>/<token>.npz "
        "(default: paths.PSEUDO_SEMANTICS_ROOT). Pass several to search them in order — Argoverse 2 needs one "
        "root per camera resolution group, see generate_pseudo_labels/README.md",
    )

    parser.add_argument("--pseudo_label_name", type=str, default="delta", help="The name of the pseudo label to save")

    parser.add_argument(
        "--scenewise_fusion",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to fuse the pseudosegs scene-wise (true/false)",
    )

    parser.add_argument(
        "--override",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether to override existing pseudosegs (true/false)",
    )

    parser.add_argument(
        "--with_score",
        type=str,
        choices=["true", "false"],
        default="false",
        help="Whether masks have corresponding confidence scores",
    )

    parser.add_argument("--num_workers", type=int, default=16, help="Number of worker threads")

    return parser.parse_args()


def main():
    args = parse_args()

    default_pseudoseg_root = (
        paths.NUSC_PSEUDOSEG_ROOT if args.dataset == "nuscenes" else paths.AV2_PSEUDOSEG_ROOT.parent
    )

    grounding_sam_config = GroundingSAMReaderConfig()
    extra_semantics_roots: Tuple[Path, ...] = ()
    if args.semantics_root:
        grounding_sam_config.semantics_root = Path(args.semantics_root[0])
        extra_semantics_roots = tuple(Path(root) for root in args.semantics_root[1:])

    config = PseudoLabelerConfig(
        pseudoseg_root=Path(args.out_dir) if args.out_dir else default_pseudoseg_root,
        dataset=args.dataset,
        split=args.split,
        prefix="-mini" if args.version == "v1.0-mini" else "",
        infos_pkl=Path(args.infos_pkl) if args.infos_pkl else None,
        nuscenes_data=Path(args.nuscenes_data) if args.nuscenes_data else paths.NUSC_ROOT,
        argoverse_data=Path(args.argoverse_data) if args.argoverse_data else paths.AV2_ROOT,
        pseudo_label_name=args.pseudo_label_name,
        scenewise_fusion=args.scenewise_fusion.lower() == "true",
        grounding_sam_reader=grounding_sam_config,
        extra_semantics_roots=extra_semantics_roots,
        override=args.override.lower() == "true",
        with_score=args.with_score.lower() == "true",
        num_workers=args.num_workers,
    )

    print("Running Pseudo Labeler with config:")
    print(f"  pseudoseg_root: {config.pseudoseg_root}")
    print(f"  split: {config.split}")
    print(f"  dataset: {config.dataset}")
    print(f"  version: {args.version}")
    print(f"  infos_pkl: {config.infos_pkl if config.infos_pkl else '<default>'}")
    print(f"  nuscenes_data: {config.nuscenes_data}")
    print(f"  argoverse_data: {config.argoverse_data}")
    print(f"  semantics_root: {grounding_sam_config.semantics_root}")
    if extra_semantics_roots:
        print(f"  extra_semantics_roots: {[str(root) for root in extra_semantics_roots]}")
    print(f"  pseudo_label_name: {config.pseudo_label_name}")
    print(f"  scenewise_fusion: {config.scenewise_fusion}")
    print(f"  override: {config.override}")
    print(f"  with_score: {config.with_score}")

    engine = PseudoLabelerEngine(config)
    engine.run()


if __name__ == "__main__":
    main()
