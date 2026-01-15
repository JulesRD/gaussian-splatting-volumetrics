import math
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np


class Camera(nn.Module):
    def __init__(self, width: int, height: int, fov_x: float, fov_y: float = None):
        super().__init__()
        self.width = width
        self.height = height
        self.fov_x = fov_x
        self.fov_y = fov_y if fov_y is not None else fov_x * height / width
        
        self.register_buffer('R', torch.eye(3, dtype=torch.float32))
        self.register_buffer('T', torch.zeros(3, dtype=torch.float32))
        
        self._update_projection()
    
    def _update_projection(self):
        fx = 0.5 * self.width / math.tan(self.fov_x * 0.5)
        fy = 0.5 * self.height / math.tan(self.fov_y * 0.5)
        cx = self.width * 0.5
        cy = self.height * 0.5
        
        self.register_buffer('focal', torch.tensor([fx, fy], dtype=torch.float32))
        self.register_buffer('center', torch.tensor([cx, cy], dtype=torch.float32))
    
    def set_pose(self, R: torch.Tensor, T: torch.Tensor):
        self.R.copy_(R)
        self.T.copy_(T)
    
    def world_to_camera(self, xyz: torch.Tensor) -> torch.Tensor:
        return (xyz @ self.R.T) + self.T
    
    def camera_to_screen(self, xyz_cam: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        depth = xyz_cam[:, 2].clamp(min=1e-6)
        x = (xyz_cam[:, 0] / depth) * self.focal[0] + self.center[0]
        y = (xyz_cam[:, 1] / depth) * self.focal[1] + self.center[1]
        return torch.stack([x, y], dim=-1), depth
    
    def project_points(self, xyz: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        xyz_cam = self.world_to_camera(xyz)
        return self.camera_to_screen(xyz_cam)


def project_points(xyz: torch.Tensor, camera) -> Tuple[torch.Tensor, torch.Tensor]:
    if isinstance(camera, Camera):
        return camera.project_points(xyz)
    
    device = xyz.device
    R = torch.from_numpy(camera.R).to(device) if hasattr(camera, 'R') else torch.eye(3, device=device)
    T = torch.from_numpy(camera.T).to(device) if hasattr(camera, 'T') else torch.zeros(3, device=device)
    
    xyz_cam = (xyz @ R.T) + T
    depth = xyz_cam[:, 2].clamp(min=1e-6)
    
    fx = 0.5 * camera.width / math.tan(camera.FovX * 0.5)
    fy = 0.5 * camera.height / math.tan(camera.FovY * 0.5)
    
    x = (xyz_cam[:, 0] / depth) * fx + camera.width * 0.5
    y = (xyz_cam[:, 1] / depth) * fy + camera.height * 0.5
    
    xy = torch.stack([x, y], dim=-1)
    return xy, depth
