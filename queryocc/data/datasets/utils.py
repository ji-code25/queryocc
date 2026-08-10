from typing import Dict, Optional, Tuple

import cv2
import numpy as np
import torch
from zod.data_classes.ego_motion import EgoMotion


def pad_offset_data_for_batching(
    offset_pointclouds: np.ndarray,
    offset_pointclouds_indices: np.ndarray,
    missing_rays: np.ndarray,
    sliced_oxts: EgoMotion,
    labels: Optional[Dict[str, np.ndarray]],
    n_max_offset_points: int,
    n_max_missing_rays: int,
    n_max_poses: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EgoMotion, Dict[str, np.ndarray]]:
    """Pad the variable-length offset supervision arrays to fixed sizes so that samples can be
    stacked by the default collate function at batch_size > 1.

    Points and missing rays are padded by duplicating random entries (duplicates are harmless
    since downstream samplers draw with replacement anyway). Ego poses are padded by repeating
    the last pose, or uniformly subsampled when over the cap (poses are only used for
    interpolation downstream, so thinning is safe). Assumes points and rays have already been
    truncated to their caps by the n_max_* checks in the caller.
    """
    if n_max_offset_points < 0 or n_max_missing_rays < 0:
        raise ValueError(
            "make_batchable requires finite caps so all samples pad to the same size: set "
            "n_max_offset_points_to_load_per_scan and n_max_missing_rays_to_load_per_scan to values >= 0."
        )
    labels = labels if labels is not None else {}

    num_points = offset_pointclouds.shape[0]
    if num_points < n_max_offset_points:
        extra = np.random.choice(num_points, n_max_offset_points - num_points, replace=True)
        offset_pointclouds = np.concatenate([offset_pointclouds, offset_pointclouds[extra]], axis=0)
        offset_pointclouds_indices = np.concatenate(
            [offset_pointclouds_indices, offset_pointclouds_indices[extra]], axis=0
        )
        for key, value in labels.items():
            labels[key] = np.concatenate([value, value[extra]], axis=0)

    num_rays = missing_rays.shape[0]
    if num_rays < n_max_missing_rays:
        extra = np.random.choice(num_rays, n_max_missing_rays - num_rays, replace=True)
        missing_rays = np.concatenate([missing_rays, missing_rays[extra]], axis=0)

    num_poses = sliced_oxts.timestamps.shape[0]
    if num_poses > n_max_poses:
        keep = np.linspace(0, num_poses - 1, n_max_poses).astype(np.int64)
        sliced_oxts.poses = sliced_oxts.poses[keep]
        sliced_oxts.timestamps = sliced_oxts.timestamps[keep]
    elif num_poses < n_max_poses:
        num_pad = n_max_poses - num_poses
        sliced_oxts.poses = np.concatenate([sliced_oxts.poses, sliced_oxts.poses[-1:].repeat(num_pad, axis=0)], axis=0)
        sliced_oxts.timestamps = np.concatenate(
            [sliced_oxts.timestamps, sliced_oxts.timestamps[-1:].repeat(num_pad, axis=0)], axis=0
        )

    return offset_pointclouds, offset_pointclouds_indices, missing_rays, sliced_oxts, labels


def crop_images(images, crop_size: Tuple[int, int, int, int]):
    """
    Args:
        images: The images to crop. (num_images, channels, height, width)
        crop_size: Tuple[int, int, int, int] = (top, right, bottom, left)
    """
    return images[..., crop_size[0] : -crop_size[2] or None, crop_size[3] : -crop_size[1] or None]


def crop_intrinsics(intrinsics: np.ndarray, crop_size: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop the intrinsics.

    Args:
        intrinsics: The intrinsics to crop.
        crop_size: The crop size.

    Returns:
        The cropped intrinsics.
    """
    out_intrinsics = intrinsics.copy()
    out_intrinsics[..., 0, 2] -= crop_size[3]
    out_intrinsics[..., 1, 2] -= crop_size[0]
    return out_intrinsics


def resize_images_with_scale(images, scalex, scaley=None):
    images = images.transpose(0, 2, 3, 1)
    images = np.array(
        [cv2.resize(image, (0, 0), fx=scalex, fy=scaley or scalex, interpolation=cv2.INTER_LINEAR) for image in images]
    )
    images = images.transpose(0, 3, 1, 2)
    return images


def resize_intrinsics(intrinsics, new_size, old_size):
    """Resize the intrinsics.

    Args:
        intrinsics: The intrinsics to resize.
        new_size: The new size of the image.

    Returns:
        The resized intrinsics.
    """
    old_width, old_height = old_size
    new_height, new_width = new_size

    scale_height = new_height / old_height
    scale_width = new_width / old_width

    out_intrinsics = intrinsics.copy()
    out_intrinsics[0, 0] *= scale_width
    out_intrinsics[1, 1] *= scale_height
    out_intrinsics[0, 2] *= scale_width
    out_intrinsics[1, 2] *= scale_height

    return out_intrinsics


def resize_intrinsics_with_scale(intrinsics, scale_width, scale_height=None):
    if isinstance(intrinsics, torch.Tensor):
        out_intrinsics = intrinsics.clone()
    else:
        out_intrinsics = intrinsics.copy()
    out_intrinsics[..., 0, 0] *= scale_width
    out_intrinsics[..., 1, 1] *= scale_height or scale_width
    out_intrinsics[..., 0, 2] *= scale_width
    out_intrinsics[..., 1, 2] *= scale_height or scale_width

    return out_intrinsics
