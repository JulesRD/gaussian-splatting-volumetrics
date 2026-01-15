import torch
import torch.nn as nn
import math
from src.rendering.camera import project_points


class Rasterizer(nn.Module):
    def __init__(self):
        super().__init__()
    
    def sort_gaussians_by_depth(self, gaussians, camera):
        xy, depth = project_points(gaussians.get_xyz, camera)
        sorted_idx = torch.argsort(depth)
        return sorted_idx, xy[sorted_idx], depth[sorted_idx]
    
    def compute_2d_covariance(self, gaussians, camera, sorted_idx):
        xyz = gaussians.get_xyz[sorted_idx]
        scaling = gaussians.get_scaling[sorted_idx]
        rotation = gaussians.get_rotation[sorted_idx]
        
        xyz_cam = camera.world_to_camera(xyz) if hasattr(camera, 'world_to_camera') else xyz
        depth = xyz_cam[:, 2].clamp(min=1e-6)
        
        focal = camera.focal if hasattr(camera, 'focal') else torch.tensor([
            0.5 * camera.width / math.tan(getattr(camera, 'FovX', math.radians(60)) * 0.5),
            0.5 * camera.height / math.tan(getattr(camera, 'FovY', math.radians(60)) * 0.5)
        ], device=xyz.device)
        
        J = torch.zeros((xyz.shape[0], 2, 3), device=xyz.device)
        J[:, 0, 0] = focal[0] / depth
        J[:, 1, 1] = focal[1] / depth
        J[:, 0, 2] = -focal[0] * xyz_cam[:, 0] / (depth * depth)
        J[:, 1, 2] = -focal[1] * xyz_cam[:, 1] / (depth * depth)
        
        s = scaling
        r = rotation
        
        q_w, q_x, q_y, q_z = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
        
        R = torch.zeros((xyz.shape[0], 3, 3), device=xyz.device)
        R[:, 0, 0] = 1 - 2 * (q_y * q_y + q_z * q_z)
        R[:, 0, 1] = 2 * (q_x * q_y - q_w * q_z)
        R[:, 0, 2] = 2 * (q_x * q_z + q_w * q_y)
        R[:, 1, 0] = 2 * (q_x * q_y + q_w * q_z)
        R[:, 1, 1] = 1 - 2 * (q_x * q_x + q_z * q_z)
        R[:, 1, 2] = 2 * (q_y * q_z - q_w * q_x)
        R[:, 2, 0] = 2 * (q_x * q_z - q_w * q_y)
        R[:, 2, 1] = 2 * (q_y * q_z + q_w * q_x)
        R[:, 2, 2] = 1 - 2 * (q_x * q_x + q_y * q_y)
        
        S = torch.diag_embed(s)
        M = R @ S
        Sigma_3d = M @ M.transpose(-1, -2)
        
        Sigma_2d = J @ Sigma_3d @ J.transpose(-1, -2)
        
        det = Sigma_2d[:, 0, 0] * Sigma_2d[:, 1, 1] - Sigma_2d[:, 0, 1] * Sigma_2d[:, 1, 0]
        det = torch.clamp(det, min=1e-6)
        
        conic = torch.zeros_like(Sigma_2d)
        conic[:, 0, 0] = Sigma_2d[:, 1, 1] / det
        conic[:, 1, 1] = Sigma_2d[:, 0, 0] / det
        conic[:, 0, 1] = -Sigma_2d[:, 0, 1] / det
        conic[:, 1, 0] = -Sigma_2d[:, 1, 0] / det
        
        return Sigma_2d, conic
    
    def rasterize(self, gaussians, camera, bg_color):
        sorted_idx, xy_sorted, depth_sorted = self.sort_gaussians_by_depth(gaussians, camera)
        
        device = gaussians.get_xyz.device
        H, W = camera.height, camera.width
        
        image = torch.zeros((3, H, W), device=device)
        alpha = torch.zeros((H, W), device=device)
        
        features = gaussians.get_features[sorted_idx]
        colors = features[:, 0, :]
        opacity = gaussians.get_opacity.squeeze(-1)[sorted_idx]
        scaling = gaussians.get_scaling[sorted_idx]
        radius = scaling.mean(dim=1) * 100.0
        
        conic = None
        
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        
        for i in range(xy_sorted.shape[0]):
            x0, y0 = xy_sorted[i]
            r = radius[i] if radius.dim() > 0 else radius
            
            if r < 1e-3:
                continue
            
            xmin = int(max(0, x0 - r))
            xmax = int(min(W - 1, x0 + r))
            ymin = int(max(0, y0 - r))
            ymax = int(min(H - 1, y0 + r))
            
            if xmin >= xmax or ymin >= ymax:
                continue
            
            dx = grid_x[ymin:ymax, xmin:xmax] - x0
            dy = grid_y[ymin:ymax, xmin:xmax] - y0
            
            if conic is not None:
                d = torch.stack([dx.flatten(), dy.flatten()], dim=1)
                weight = torch.exp(-0.5 * torch.sum(d * (d @ conic[i]), dim=1))
                weight = weight.reshape(dy.shape)
            else:
                weight = torch.exp(-0.5 * ((dx / r) ** 2 + (dy / r) ** 2))
            
            a = opacity[i] * weight
            one_minus_alpha = 1.0 - alpha[ymin:ymax, xmin:xmax]
            contrib = a * one_minus_alpha
            
            if colors[i].numel() == 3:
                color_contrib = colors[i].view(3, 1, 1) * contrib
            else:
                color_contrib = colors[i].view(-1, 1, 1).expand(3, contrib.shape[0], contrib.shape[1]) * contrib
            
            image[:, ymin:ymax, xmin:xmax] += color_contrib
            alpha[ymin:ymax, xmin:xmax] += contrib
        
        image += bg_color.view(3, 1, 1) * (1.0 - alpha)
        return image.clamp(0.0, 1.0)


def gaussian_2d(dx: torch.Tensor, dy: torch.Tensor, sigma_x: torch.Tensor, sigma_y: torch.Tensor):
    return torch.exp(-0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2))


def render_gaussians(gaussians, camera, bg_color: torch.Tensor):
    rasterizer = Rasterizer()
    return rasterizer.rasterize(gaussians, camera, bg_color)


if __name__ == "__main__":
    import math
    from src.rendering.camera import Camera
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    class DummyGaussians:
        def __init__(self):
            self.xyz = torch.tensor([[0.0, 0.0, 2.0]], device=device)
            self.scale = torch.tensor([[0.05, 0.05, 0.05]], device=device)
            self.color = torch.tensor([[[1.0], [0.0], [0.0]]], device=device)
            self.opacity = torch.tensor([[0.8]], device=device)
            self.rot = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device)
        
        @property
        def get_xyz(self): return self.xyz
        @property
        def get_scaling(self): return self.scale
        @property
        def get_features(self): return self.color
        @property
        def get_opacity(self): return self.opacity
        @property
        def get_rotation(self): return self.rot
    
    gaussians = DummyGaussians()
    camera = Camera(256, 256, math.radians(60))
    bg = torch.zeros(3, device=device)
    
    img = render_gaussians(gaussians, camera, bg)
    print("Rasterizer test image:", img.shape, img.min().item(), img.max().item())
