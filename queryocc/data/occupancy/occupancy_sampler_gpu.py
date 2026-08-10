from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch import Tensor

import queryocc.config.base_config as cfg
from queryocc.common.types import PointCloudRange
from queryocc.data.datasets.nuscenes_mappings import LabelMapper
from queryocc.data.utils.ray_intersection import calculate_max_ray_distances_pt
from queryocc.model.ops.functions.interpolate import interpolate_trajectories


def _transform(points: Tensor, poses: Tensor) -> Tensor:
    return (poses[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + poses[..., :3, 3]


def _stack(input: List[Tensor]) -> Optional[Tensor]:
    if input is None or len(input) == 0:
        return None
    return torch.stack(input)


def panoptic_sampling_available(labels: Optional[Dict[str, Tensor]]) -> bool:
    do_panoptic_sampling = False
    panoptic_valid = labels.get("panoptic_valid", None)
    if panoptic_valid is not None:
        # Check if all batches have any valid panoptic labels
        do_panoptic_sampling = panoptic_valid.any(dim=1).all().item()
    return do_panoptic_sampling


@dataclass
class LidarGPUOccupancySamplerConfig(cfg.InstantiateConfig):
    _target: Type = field(default_factory=lambda: LidarGPUOccupancySampler)

    num_lidar_rays: int = 100_000
    """Number of lidar points to sample (* batchsize)."""
    num_missing_rays: int = 20_000
    """Number of missing rays to sample (* batchsize)."""
    num_query_points: int = 900_000
    """Number of positive samples to generate."""

    buffer_size: float = 0.1
    """Buffer size behind the positive samples."""

    negative_bias_power: Union[int, float] = 1
    """Whether to bias the negatives to be close to positives by taking (jitter)^power. 1 is no bias."""
    negative_from_points_ratio: float = -1.0  # [0.0, 1.0] for static ratio, -1.0 for dynamic ratio
    """Ratio of negative samples to generate from points in bev range (vs negative directions)."""
    negative_samples_based_on_ray_length: bool = True
    """Whether to sample negatives based on the length of the ray. This effectively samples more queries from longer rays."""

    positive_samples_based_on: Literal[
        "ray_length",
        "uniform",
        "grid_based",
        "grid_uniform",
        "voxel_based",
        "semantic_labels_dynamic_ratio",
        "semantic_labels_dynamic_upsample",
        "semantic_labels_global_class_statistics",
    ] = "uniform"
    """Whether to sample positives based on the length of the ray. This effectively samples more queries from longer rays."""
    positive_samples_based_on_ray_length_power: float = 1.0
    """Power to raise the ray length to. 1.0 is no power."""
    positive_samples_grid_cell_size: float = 0.50
    """Size of the grid cell in meters."""
    positive_samples_voxel_size: float = 0.50
    """Size of the voxel in meters."""
    positive_samples_voxel_size_z_boost: float = 5.0
    """Boost factor for the z dimension of the voxel."""

    dynamic_ratio: float = 0.5
    """Ratio of dynamic points to static points. 0.9 means 90% of the points are dynamic points and 10% are static points."""
    dynamic_upsample: float = 3.0
    """Upsample factor for the dynamic points. 3.0 means 3x more dynamic points than what is naturally present."""
    global_class_statistics_alpha: float = 0.5
    """Alpha for the global class statistics. 1.0 means no balancing. 0.0 means full inverse-frequency weighting, extremely aggressive."""


class LidarGPUOccupancySampler(torch.nn.Module):
    def __init__(
        self,
        config: LidarGPUOccupancySamplerConfig,
        point_cloud_range: PointCloudRange,
    ):
        super().__init__()
        self.config = config
        self.point_cloud_range = point_cloud_range

    def _uniform_subsample(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            idx = torch.randint(0, pc.shape[0], (num_per_sample,), dtype=torch.long, device=pc.device)
            subsampled_points.append(pc[idx])

            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idx])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_ray_length(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            lengths = pc[..., :3].norm(dim=-1, keepdim=False)
            probabilities = lengths**self.config.positive_samples_based_on_ray_length_power
            probabilities = probabilities / probabilities.sum(dim=0, keepdim=True)
            idxs = torch.multinomial(probabilities, num_per_sample, replacement=True)
            subsampled_points.append(pc[idxs])

            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idxs])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_grid(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        """
        Sampling points based on a grid of size 1.28m x 1.28m in the BEV.
        The probability of sampling a cell is the same for all cells.
        A random points is sampled from the cell.

        """
        # Get the grid cell size
        grid_cell_size = self.config.positive_samples_grid_cell_size

        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        # Scatter points into grid cells
        for i, pc in enumerate(points):
            min_x, min_y, min_z, _ = pc.min(dim=0).values
            max_x, max_y, max_z, _ = pc.max(dim=0).values
            num_grid_cells_x = int((max_x - min_x) / grid_cell_size)
            num_grid_cells_y = int((max_y - min_y) / grid_cell_size)
            num_grid_cells = num_grid_cells_x * num_grid_cells_y

            # Calculate grid cell indices
            points_indeces_in_grid_cells = (
                pc[..., :2] - torch.tensor([min_x, min_y], device=pc.device)
            ) / grid_cell_size
            points_indeces_in_grid_cells = points_indeces_in_grid_cells.long()

            # Ensure indices are within bounds
            points_indeces_in_grid_cells[..., 0] = torch.clamp(
                points_indeces_in_grid_cells[..., 0], 0, num_grid_cells_x - 1
            )
            points_indeces_in_grid_cells[..., 1] = torch.clamp(
                points_indeces_in_grid_cells[..., 1], 0, num_grid_cells_y - 1
            )

            # Calculate final indices
            points_indeces_in_grid_cells = (
                points_indeces_in_grid_cells[..., 0] * num_grid_cells_y + points_indeces_in_grid_cells[..., 1]
            )
            points_indeces_in_grid_cells = points_indeces_in_grid_cells.long()

            num_points_per_cell = torch.zeros(num_grid_cells, device=pc.device, dtype=torch.int64)
            num_points_per_cell = num_points_per_cell.scatter_reduce(
                0,
                points_indeces_in_grid_cells,
                torch.ones(points_indeces_in_grid_cells.shape[0], device=pc.device, dtype=torch.int64),
                reduce="sum",
            )

            # Get the probability per point conditioned on the grid cell p(point | grid_cell)
            # num_non_empty_cells = (num_points_per_cell > 0).sum()
            # p_cell = 1 / num_non_empty_cells
            p_point_given_cell = 1 / num_points_per_cell[points_indeces_in_grid_cells]
            # p_point = p_cell * p_point_given_cell
            idxs = torch.multinomial(p_point_given_cell, num_per_sample, replacement=True)
            selected_points = pc[idxs]

            subsampled_points.append(selected_points)
            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idxs])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_true_grid_uniform(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        grid_cell_size = self.config.positive_samples_grid_cell_size
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            min_xy = pc[:, :2].min(dim=0).values
            max_xy = pc[:, :2].max(dim=0).values
            num_cells_x = int((max_xy[0] - min_xy[0]) / grid_cell_size) + 1
            num_cells_y = int((max_xy[1] - min_xy[1]) / grid_cell_size) + 1

            # Compute flat grid indices per point
            grid_indices = ((pc[:, :2] - min_xy) / grid_cell_size).long()
            grid_indices[:, 0] = grid_indices[:, 0].clamp(0, num_cells_x - 1)
            grid_indices[:, 1] = grid_indices[:, 1].clamp(0, num_cells_y - 1)
            flat_indices = grid_indices[:, 0] * num_cells_y + grid_indices[:, 1]

            # Get unique cells and mapping
            unique_cells, inverse = torch.unique(flat_indices, return_inverse=True)
            num_cells = unique_cells.shape[0]

            # Sample grid cells uniformly
            sampled_cell_ids = torch.randint(0, num_cells, (num_per_sample,), device=pc.device)
            sampled_flat_ids = unique_cells[sampled_cell_ids]  # [num_per_sample]

            # Create a mapping: flat index -> all point indices (grouped)
            # Sort flat_indices to group by cell
            sorted_ids, sort_idx = torch.sort(flat_indices)
            pc_sorted = pc[sort_idx]

            # Find cell boundaries
            boundaries = torch.bucketize(sampled_flat_ids, sorted_ids, right=True)
            left = torch.searchsorted(sorted_ids, sampled_flat_ids, right=False)
            right = boundaries

            # For each sampled cell, select a random point index in [left, right)
            range_lengths = right - left
            empty_mask = range_lengths == 0
            range_lengths[empty_mask] = 1  # Avoid zero-length when sampling

            rand_offsets = torch.floor(torch.rand_like(range_lengths.float()) * range_lengths).long()
            sampled_offsets = left + rand_offsets
            sampled_offsets = sampled_offsets.clamp(0, pc_sorted.shape[0] - 1)

            selected_points = pc_sorted[sampled_offsets]
            subsampled_points.append(selected_points)
            for key, label in labels.items():
                subsampled_labels[key].append(label[i][sort_idx][sampled_offsets])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_voxel(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Dict[str, Tensor] = {},
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        for i, pc in enumerate(points):
            min_x, min_y, min_z, _ = pc.min(dim=0).values
            max_x, max_y, max_z, _ = pc.max(dim=0).values
            z_boost = self.config.positive_samples_voxel_size_z_boost

            num_voxels_x = int((max_x - min_x) / self.config.positive_samples_voxel_size)
            num_voxels_y = int((max_y - min_y) / self.config.positive_samples_voxel_size)
            num_voxels_z = int(torch.ceil((max_z - min_z) / (self.config.positive_samples_voxel_size / z_boost)))

            # Calculate voxel indices
            voxel_sizes = torch.tensor(
                [
                    self.config.positive_samples_voxel_size,
                    self.config.positive_samples_voxel_size,
                    self.config.positive_samples_voxel_size / z_boost,
                ],
                device=pc.device,
            )
            voxel_indices = ((pc[:, :3] - torch.tensor([min_x, min_y, min_z], device=pc.device)) / voxel_sizes).long()
            voxel_indices[:, 0] = voxel_indices[:, 0].clamp(0, num_voxels_x - 1)
            voxel_indices[:, 1] = voxel_indices[:, 1].clamp(0, num_voxels_y - 1)
            voxel_indices[:, 2] = voxel_indices[:, 2].clamp(0, num_voxels_z - 1)

            flat_voxel_indices = (
                voxel_indices[:, 0] * num_voxels_y * num_voxels_z
                + voxel_indices[:, 1] * num_voxels_z
                + voxel_indices[:, 2]
            )
            num_points_per_voxel = torch.zeros(
                num_voxels_x * num_voxels_y * num_voxels_z, device=pc.device, dtype=torch.int16
            )

            num_points_per_voxel = num_points_per_voxel.scatter_reduce(
                0,
                flat_voxel_indices,
                torch.ones(flat_voxel_indices.shape[0], device=pc.device, dtype=torch.int16),
                reduce="sum",
            )

            non_zero_num_points_per_voxel = num_points_per_voxel[num_points_per_voxel > 0].float()
            # Find the 90th percentile
            percentile_90 = torch.quantile(non_zero_num_points_per_voxel, 1.0)
            percentile_10 = torch.quantile(non_zero_num_points_per_voxel, 0.75)
            # Clip the values to be between the 10th and 90th percentile
            num_points_per_voxel = num_points_per_voxel.clamp(percentile_10, percentile_90)

            point_weights = 1 / num_points_per_voxel[flat_voxel_indices]

            idxs = torch.multinomial(point_weights, num_per_sample, replacement=True)
            selected_points = pc[idxs]
            subsampled_points.append(selected_points)
            for key, label in labels.items():
                subsampled_labels[key].append(label[i][idxs])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}

        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_semantic_labels_dynamic_ratio(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Optional[Dict[str, Tensor]],
        label_mapper: Optional[LabelMapper] = None,
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        lidarseg = labels.get("panoptic_lidarseg", None)
        assert lidarseg is not None, "Panoptic labels are required for semantic sampling"

        for i, (pc, label) in enumerate(zip(points, lidarseg)):
            dynamic_mask = label_mapper.learning_to_dynamic(label)

            static_mask = label_mapper.learning_to_static(label)
            free_mask = label_mapper.learning_to_free(label)
            noise_mask = label_mapper.learning_to_noise(label)

            # Simplifying to dynamic vs static (free and noise are included in static)
            dynamic_mask = dynamic_mask & ~free_mask & ~noise_mask
            static_mask = static_mask | free_mask | noise_mask

            num_dynamic_points_to_sample = int(num_per_sample * self.config.dynamic_ratio)
            num_static_points_to_sample = num_per_sample - num_dynamic_points_to_sample

            groups = {
                "dynamic": torch.where(dynamic_mask)[0],
                "static": torch.where(static_mask)[0],
            }
            target_counts = {
                "dynamic": num_dynamic_points_to_sample,
                "static": num_static_points_to_sample,
            }

            sampled_indices = []
            for group, indices in groups.items():
                n = target_counts[group]
                if n == 0 or len(indices) == 0:
                    continue
                if n <= len(indices):
                    # Sample without replacement
                    perm = torch.randperm(len(indices), device=indices.device)
                    chosen = indices[perm[:n]]
                else:
                    # Not enough points → sample with replacement
                    chosen = indices[torch.randint(len(indices), (n,), device=indices.device)]
                sampled_indices.append(chosen)

            if len(sampled_indices) == 0:
                # fallback: uniform random from all points
                all_indices = torch.arange(len(pc), device=pc.device)
                chosen = all_indices[torch.randint(len(all_indices), (num_per_sample,), device=pc.device)]
                sampled_indices = [chosen]

            sampled_indices = torch.cat(sampled_indices, dim=0)

            subsampled_points.append(pc[sampled_indices])
            for k, lbl in labels.items():
                subsampled_labels[k].append(lbl[i][sampled_indices])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}
        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_semantic_labels_dynamic_upsample(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Optional[Dict[str, Tensor]],
        label_mapper: Optional[LabelMapper] = None,
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        lidarseg = labels.get("panoptic_lidarseg", None)
        assert lidarseg is not None, "Panoptic labels are required for semantic sampling"

        for i, (pc, label) in enumerate(zip(points, lidarseg)):
            dynamic_mask = label_mapper.learning_to_dynamic(label)
            static_mask = label_mapper.learning_to_static(label)
            free_mask = label_mapper.learning_to_free(label)
            noise_mask = label_mapper.learning_to_noise(label)

            # Simplify: dynamic vs static (free + noise grouped into static)
            dynamic_mask = dynamic_mask & ~free_mask & ~noise_mask
            static_mask = static_mask | free_mask | noise_mask

            # Build weights
            weights = torch.ones(len(pc), device=pc.device, dtype=torch.float32)
            weights[dynamic_mask.squeeze()] *= self.config.dynamic_upsample  # e.g. 3.0× boost

            # Normalize so they sum to 1
            weights = weights / weights.sum()

            # Weighted random choice (with replacement, which is standard in this context)
            chosen = torch.multinomial(weights, num_samples=num_per_sample, replacement=True)

            subsampled_points.append(pc[chosen])
            for k, lbl in labels.items():
                subsampled_labels[k].append(lbl[i][chosen])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}
        return _stack(subsampled_points), subsampled_labels

    def _subsample_based_on_semantic_labels_global_class_statistics(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Optional[Dict[str, Tensor]],
        label_mapper: Optional[LabelMapper] = None,
    ) -> Tuple[Optional[Tensor], Optional[Dict[str, Tensor]]]:
        """
        Sample points per frame based on global class statistics. Rare classes are more likely to be sampled.
        """
        subsampled_points = []
        subsampled_labels = {k: [] for k in labels.keys()}

        lidarseg = labels.get("panoptic_lidarseg", None)
        assert lidarseg is not None, "Panoptic labels are required for semantic sampling"

        # Get global class counts: tensor of shape (num_classes,)
        num_points_per_class = label_mapper.learning_to_num_points(lidarseg)  # shape: [bs, num_classes, 1]
        class_weights = 1.0 / (num_points_per_class.float() ** self.config.global_class_statistics_alpha)
        class_weights[num_points_per_class == 0] = 0.0  # avoid divide by zero for classes with zero points

        for i, (pc, label, weights) in enumerate(zip(points, lidarseg, class_weights)):
            valid_mask = weights > 0
            valid_weights = weights[valid_mask]
            valid_indices = torch.nonzero(valid_mask.squeeze(), as_tuple=False).squeeze(1)
            sampled_indices = valid_indices[torch.multinomial(valid_weights, num_per_sample, replacement=True)]
            subsampled_points.append(pc[sampled_indices])
            for k, lbl in labels.items():
                subsampled_labels[k].append(lbl[i][sampled_indices])

        subsampled_labels = {k: _stack(v) for k, v in subsampled_labels.items()}
        return _stack(subsampled_points), subsampled_labels

    def _subsample(
        self,
        points: Tensor,
        num_per_sample: int,
        labels: Optional[Dict[str, Tensor]],
        label_mapper: Optional[LabelMapper] = None,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        if self.training:
            if self.config.positive_samples_based_on == "ray_length":
                return self._subsample_based_on_ray_length(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "uniform":
                return self._uniform_subsample(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "grid_based":
                return self._subsample_based_on_grid(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "grid_uniform":
                return self._subsample_true_grid_uniform(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "voxel_based":
                return self._subsample_based_on_voxel(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "semantic_labels_dynamic_ratio":
                if panoptic_sampling_available(labels):
                    return self._subsample_based_on_semantic_labels_dynamic_ratio(
                        points, num_per_sample, labels, label_mapper
                    )
                else:
                    return self._subsample_based_on_ray_length(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "semantic_labels_dynamic_upsample":
                if panoptic_sampling_available(labels):
                    return self._subsample_based_on_semantic_labels_dynamic_upsample(
                        points, num_per_sample, labels, label_mapper
                    )
                else:
                    return self._subsample_based_on_ray_length(points, num_per_sample, labels)
            elif self.config.positive_samples_based_on == "semantic_labels_global_class_statistics":
                if panoptic_sampling_available(labels):
                    return self._subsample_based_on_semantic_labels_global_class_statistics(
                        points, num_per_sample, labels, label_mapper
                    )
                else:
                    return self._subsample_based_on_ray_length(points, num_per_sample, labels)
            else:
                raise ValueError(f"Invalid positive samples based on: {self.config.positive_samples_based_on}")
        else:
            return self._uniform_subsample(points, num_per_sample, labels)

    def _compensate(
        self, points: Tensor, ego_poses: Tensor, ego_times: Tensor, extrinsics: Tensor, t_idx: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Compute compensated points and point origins (to t=0).

        extrinsics is either a single sensor pose per sample (B, 4, 4), or per-timestep sensor poses
        (B, T, 4, 4) selected per point via t_idx (B, N) when each point may come from a different camera.
        """
        poses = interpolate_trajectories(ego_poses, ego_times, points[..., -1].contiguous())  # B, N, 3, 4
        if extrinsics.dim() == 4:
            assert t_idx is not None, "t_idx is required to select per-point extrinsics when they are per-timestep"
            batch_idx = torch.arange(extrinsics.shape[0], device=extrinsics.device).unsqueeze(-1)
            point_extrinsics = extrinsics[batch_idx, t_idx]  # B, N, 4, 4
        else:
            point_extrinsics = extrinsics.unsqueeze(1)
        lidar2target = (poses @ point_extrinsics).to(points.dtype)  # B, N, 4, 4
        points[..., :3] = _transform(points[..., :3], lidar2target)
        origins = lidar2target[..., :3, 3]
        return points, origins

    def forward(self, *args, **kwargs):
        return self.sample(*args, **kwargs)

    @torch.no_grad()
    def sample(
        self,
        positive_points: Tensor,  # variable per sample
        ego_poses: Tensor,
        ego_times: Tensor,
        extrinsics: Tensor,
        negative_rays: Tensor,  # variable per sample
        future_points_ratio_inside: Tensor,
        positive_point_labels: Dict[str, Tensor],
        label_mapper: Optional[LabelMapper] = None,
        positive_t_idx: Optional[Tensor] = None,  # B, N timestep index per point, for (B, T, 4, 4) extrinsics
        negative_rays_t_idx: Optional[Tensor] = None,  # B, M timestep index per missing ray
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Sample occupancy queries and labels."""
        bs, device = positive_points.shape[0], positive_points.device

        num_positive_points = positive_points.shape[1]
        if num_positive_points == 0:
            return (
                torch.empty((bs, 0, 4), device=device),
                torch.empty((bs, 0, 1), device=device),
                {k: torch.empty_like(v) for k, v in positive_point_labels.items()},
            )

        # ride the labels dict through _subsample so the timestep indices stay aligned with the points
        if positive_t_idx is not None:
            positive_point_labels = {**positive_point_labels, "_t_idx": positive_t_idx}

        points, labels = self._subsample(
            positive_points, self.config.num_lidar_rays, positive_point_labels, label_mapper
        )
        t_idx = labels.pop("_t_idx", None)

        # compensate to ego frame
        points, origins = self._compensate(points, ego_poses, ego_times, extrinsics, t_idx)

        # compute lengths and directions
        unnorm_directions = points[..., :3] - origins
        lengths = unnorm_directions.norm(dim=-1, keepdim=True)
        # clamp_min guards against a point coinciding with its ray origin (0/0 -> NaN), e.g. a
        # pseudo point at depth ~0. lengths itself (used for the negative-query offset below) is
        # left unclamped so a true zero-length ray contributes a zero-length offset.
        directions = unnorm_directions / lengths.clamp_min(1e-6)

        # sample num_query_points positive queries from points in range
        pos_idxs = torch.randint(0, points.shape[1], (bs, self.config.num_query_points), device=device)
        batch_idx = torch.arange(bs, device=device).unsqueeze(-1)
        positive_queries = points[batch_idx, pos_idxs]

        # Create positive queries
        jitter = torch.rand((bs, self.config.num_query_points, 1), device=device)
        positive_queries[..., :3] += directions[batch_idx, pos_idxs] * jitter * self.config.buffer_size

        # needs to be computed on the fly to handle case where negative rays are not available
        negative_from_points_ratio = self._compute_negative_from_points_ratio(future_points_ratio_inside)
        num_neg_from_points = self._num_neg_from_points(negative_rays, negative_from_points_ratio)
        num_neg_from_rays = self.config.num_query_points - num_neg_from_points

        # sample negative queries from positive points
        if self.config.negative_samples_based_on_ray_length:
            probabilities = lengths.squeeze(-1)
            probabilities = probabilities / probabilities.sum(dim=1, keepdim=True)
            neg_idxs = torch.multinomial(probabilities, num_neg_from_points, replacement=True)
        else:
            neg_idxs = torch.randint(0, points.shape[1], (bs, num_neg_from_points), device=device)

        negative_queries = points[batch_idx, neg_idxs]
        jitter = torch.rand((bs, num_neg_from_points, 1), device=device)
        if self.config.negative_bias_power != 1:
            jitter **= self.config.negative_bias_power
        negative_queries[..., :3] -= directions[batch_idx, neg_idxs] * jitter * lengths[batch_idx, neg_idxs]

        # sample negative queries from missing rays
        if num_neg_from_rays:
            neg_ray_queries = self._sample_from_missing_rays(
                negative_rays, num_neg_from_rays, ego_poses, ego_times, extrinsics, negative_rays_t_idx
            )
            negative_queries = torch.cat([negative_queries, neg_ray_queries], dim=1)

        # concatenate positive and negative queries
        all_queries = torch.cat([positive_queries, negative_queries], dim=1)
        all_occupancy = torch.cat(
            [
                torch.ones((bs, self.config.num_query_points, 1), device=device, dtype=torch.float32),
                torch.zeros((bs, self.config.num_query_points, 1), device=device, dtype=torch.float32),
            ],
            dim=1,
        )

        all_labels = {}
        for key, label in labels.items():
            if "rgb" in key:
                # Float regression target (per-point colour); never cast to long. Only positive
                # (occupied) queries carry a meaningful colour, so the negatives are filled with a
                # placeholder that the decoder loss masks out via the occupancy label.
                positive_labels = label[batch_idx, pos_idxs]
                negative_labels = torch.zeros_like(positive_labels)
                all_labels[key] = torch.cat([positive_labels, negative_labels], dim=1)
                continue
            positive_labels = label[batch_idx, pos_idxs].long()
            if "valid" in key:
                negative_labels = torch.ones_like(positive_labels)
            elif "vfm" in key:
                negative_labels = torch.zeros_like(positive_labels)
            else:
                negative_labels = (
                    torch.ones((bs, self.config.num_query_points, 1), device=device, dtype=torch.long) * -1
                )
            all_labels[key] = torch.cat([positive_labels, negative_labels], dim=1)

        return all_queries, all_occupancy, all_labels

    def _num_neg_from_points(self, negative_rays: Tensor, negative_from_points_ratio: float) -> int:
        """Determine ratio of negative queries from positive points vs missing rays."""
        if len(negative_rays) and negative_from_points_ratio < 1.0:
            if not all((neg_rays.shape[0] for neg_rays in negative_rays)):
                # special case: if no negative rays are available (for any sample in the batch), sample all negatives from points
                num_neg_from_points = self.config.num_query_points
            else:
                num_neg_from_points = int(self.config.num_query_points * negative_from_points_ratio)
        else:
            num_neg_from_points = self.config.num_query_points
        return num_neg_from_points

    def _compute_negative_from_points_ratio(self, future_points_ratio_inside: Tensor) -> float:
        """Compute the ratio of negative queries from positive points vs missing rays."""
        if self.config.negative_from_points_ratio < 0.0:
            # We need to have a ratio that is the same for all samples in the batch, so we take the mean
            ratio = future_points_ratio_inside.sum() / future_points_ratio_inside.shape[0]
        else:
            ratio = self.config.negative_from_points_ratio
        return ratio

    def _sample_from_missing_rays(
        self,
        negative_rays: Tensor,
        num_queries: int,
        ego_poses: Tensor,
        ego_times: Tensor,
        extrinsics: Tensor,
        t_idx: Optional[Tensor] = None,
    ) -> Tensor:
        """Sample negative queries from missing rays."""
        num_rays = min(self.config.num_missing_rays, num_queries)
        # TODO: Verify that it's better to do uniform subsampling here rather than a potentially dynamic one...
        if t_idx is not None:
            neg_rays, neg_ray_labels = self._uniform_subsample(negative_rays, num_rays, {"_t_idx": t_idx})
            neg_t_idx = neg_ray_labels["_t_idx"]
        else:
            neg_rays = self._uniform_subsample(negative_rays, num_rays)[0]
            neg_t_idx = None
        device, bs = neg_rays.device, neg_rays.shape[0]
        batch_idx = torch.arange(bs, device=device).unsqueeze(-1)

        neg_rays, neg_origins = self._compensate(neg_rays, ego_poses, ego_times, extrinsics, neg_t_idx)
        neg_rays[..., :3] = neg_rays[..., :3] - neg_origins
        # clamp_min guards against a ray point coinciding with its origin (0/0 -> NaN).
        neg_rays[..., :3] = neg_rays[..., :3] / neg_rays[..., :3].norm(dim=-1, keepdim=True).clamp_min(1e-6)

        neg_lengths = calculate_max_ray_distances_pt(
            origins=neg_origins,
            normalized_directions=neg_rays[..., :3],
            bounds_min=torch.tensor(self.point_cloud_range[:3], device=device),
            bounds_max=torch.tensor(self.point_cloud_range[3:], device=device),
        ).unsqueeze(-1)

        neg_idxs = torch.randint(0, neg_rays.shape[1], (bs, num_queries), device=device)
        jitter = torch.rand((bs, num_queries, 1), device=device)
        neg_ray_queries = neg_rays[batch_idx, neg_idxs]
        neg_ray_queries[..., :3] *= neg_lengths[batch_idx, neg_idxs] * jitter
        neg_ray_queries[..., :3] += neg_origins[batch_idx, neg_idxs]

        return neg_ray_queries


if __name__ == "__main__":
    from queryocc.model.base import DEFAULT_PCD_RANGE_FULL

    sampler = LidarGPUOccupancySamplerConfig().setup(point_cloud_range=DEFAULT_PCD_RANGE_FULL)
    sampler.sample(
        positive_points=torch.rand((2, 1000, 4)),
        ego_poses=torch.rand((2, 100, 4, 4)),
        ego_times=(torch.arange(0, 100, 1) / 100).unsqueeze(0).repeat(2, 1),
        extrinsics=torch.eye(4).unsqueeze(0).repeat(2, 1, 1),
        negative_rays=None,
    )
