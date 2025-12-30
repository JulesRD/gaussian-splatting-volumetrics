import math
from typing import Tuple

import torch


def project_points(
    xyz: torch.Tensor,
    camera,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Projects 3D points to 2D screen space.

    Args:
        xyz: (N, 3) world-space points
        camera: CameraInfo

    Returns:
        xy: (N, 2) pixel coordinates
        depth: (N,) depth values
    """
    device = xyz.device

    R = torch.from_numpy(camera.R).to(device)
    T = torch.from_numpy(camera.T).to(device)

    # World → camera
    xyz_cam = (xyz @ R.T) + T

    depth = xyz_cam[:, 2].clamp(min=1e-6)

    # Perspective projection
    fx = 0.5 * camera.width / math.tan(camera.FovX * 0.5)
    fy = 0.5 * camera.height / math.tan(camera.FovY * 0.5)

    x = (xyz_cam[:, 0] / depth) * fx + camera.width * 0.5
    y = (xyz_cam[:, 1] / depth) * fy + camera.height * 0.5

    xy = torch.stack([x, y], dim=-1)
    return xy, depth
