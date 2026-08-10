import torch


def compute_bbox_corners_2d(centers, sizes, rotations):
    """
    Computes the corner coordinates for a 2D bounding box.
    """
    num_objects = centers.shape[0]
    x = torch.tensor([-0.5, 0.5])
    y = torch.tensor([-0.5, 0.5])
    xx, yy = torch.meshgrid(x, y)
    pos = torch.stack((xx.flatten(), yy.flatten()), dim=-1)
    num_points = pos.shape[0]
    # Scale
    pos = pos.unsqueeze(0).expand(centers.shape[0], -1, -1)
    pos = pos.to(centers.device)
    pos = pos * sizes[:, :2].unsqueeze(1)  # num_objects, num_points, 2
    # Make homogeneous
    pos = torch.cat((pos, torch.ones_like(pos[..., :1])), dim=-1)  # num_objects, num_points, 3

    # Create transformation matrix
    rotations = rotations.unsqueeze(1).expand(-1, num_points, -1, -1)  # num_objects, num_points, 2, 2
    translations = centers.unsqueeze(1).expand(-1, num_points, -1)  # num_objects, num_points, 2

    # pre allocate transformation matrix # num_objects, num_points, 3, 3
    transformations = torch.zeros(rotations.shape[0], rotations.shape[1], 3, 3, device=centers.device, dtype=pos.dtype)
    transformations[:, :, :2, :2] = rotations[:, :, :2, :2]
    transformations[:, :, :2, 2] = translations[:, :, :2]
    transformations[:, :, 2, 2] = 1.0

    # flatten for matrix multiplication
    transformations = transformations.reshape(-1, 3, 3)
    pos = pos.reshape(-1, 3)

    # transform
    pos = transformations @ pos[:, :, None]

    # unflatten
    pos = pos.reshape(num_objects, num_points, 3, 1)
    pos = pos[:, :, :2, 0]

    return pos
