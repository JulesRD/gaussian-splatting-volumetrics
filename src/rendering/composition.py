import torch
import torch.nn as nn
from typing import List, Tuple, Optional


class AlphaCompositor(nn.Module):
    def __init__(self, blend_mode="front-to-back"):
        super().__init__()
        self.blend_mode = blend_mode
    
    def front_to_back_blend(self, colors: torch.Tensor, alphas: torch.Tensor, 
                            depths: torch.Tensor, bg_color: torch.Tensor):
        device = colors.device
        N = colors.shape[0]
        H, W = alphas.shape[1], alphas.shape[2]
        
        sorted_idx = torch.argsort(depths)
        colors_sorted = colors[sorted_idx]
        alphas_sorted = alphas[sorted_idx]
        
        image = bg_color.view(3, 1, 1).expand(3, H, W).clone()
        transmittance = torch.ones((H, W), device=device)
        
        for i in range(N):
            alpha_i = alphas_sorted[i]
            color_i = colors_sorted[i]
            
            contribution = alpha_i * transmittance
            image += color_i.view(3, 1, 1) * contribution
            transmittance = transmittance * (1.0 - alpha_i)
            
            if transmittance.max() < 0.001:
                break
        
        return image.clamp(0, 1)
    
    def back_to_front_blend(self, colors: torch.Tensor, alphas: torch.Tensor,
                            depths: torch.Tensor, bg_color: torch.Tensor):
        device = colors.device
        N = colors.shape[0]
        H, W = alphas.shape[1], alphas.shape[2]
        
        sorted_idx = torch.argsort(depths, descending=True)
        colors_sorted = colors[sorted_idx]
        alphas_sorted = alphas[sorted_idx]
        
        image = bg_color.view(3, 1, 1).expand(3, H, W).clone()
        
        for i in range(N):
            alpha_i = alphas_sorted[i]
            color_i = colors_sorted[i]
            
            image = color_i.view(3, 1, 1) * alpha_i + image * (1.0 - alpha_i)
        
        return image.clamp(0, 1)
    
    def forward(self, colors: torch.Tensor, alphas: torch.Tensor,
                depths: torch.Tensor, bg_color: torch.Tensor):
        if self.blend_mode == "front-to-back":
            return self.front_to_back_blend(colors, alphas, depths, bg_color)
        else:
            return self.back_to_front_blend(colors, alphas, depths, bg_color)


class SurfaceVolumeCompositor(nn.Module):
    def __init__(self):
        super().__init__()
        self.alpha_compositor = AlphaCompositor(blend_mode="front-to-back")
    
    def compute_volumetric_contribution(self, volume_gaussians, camera, 
                                       xy_sorted: torch.Tensor, 
                                       depth_sorted: torch.Tensor,
                                       H: int, W: int, device):
        density = volume_gaussians.get_density[depth_sorted.long()]
        phase_color = volume_gaussians.get_phase_color[depth_sorted.long()]
        opacity = volume_gaussians.get_opacity.squeeze(-1)[depth_sorted.long()]
        scaling = volume_gaussians.get_scaling[depth_sorted.long()]
        
        radius = scaling.mean(dim=1) * 100.0
        
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        
        volume_alpha = torch.zeros((H, W), device=device)
        volume_color = torch.zeros((3, H, W), device=device)
        transmittance = torch.ones((H, W), device=device)
        
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
            
            weight = torch.exp(-0.5 * ((dx / r) ** 2 + (dy / r) ** 2))
            
            alpha_vol = density[i] * weight * opacity[i]
            alpha_vol = alpha_vol.clamp(0, 1)
            
            contrib = alpha_vol * transmittance[ymin:ymax, xmin:xmax]
            
            volume_alpha[ymin:ymax, xmin:xmax] += contrib
            volume_color[:, ymin:ymax, xmin:xmax] += (
                phase_color[i].view(3, 1, 1) * contrib
            )
            
            transmittance[ymin:ymax, xmin:xmax] *= (1.0 - alpha_vol)
        
        return volume_color, volume_alpha
    
    def composite_scene(self, scene, camera, bg_color):
        device = scene.device
        H, W = camera.height, camera.width
        
        final_image = bg_color.view(3, 1, 1).expand(3, H, W).clone()
        transmittance = torch.ones((H, W), device=device)
        
        all_gaussians = []
        all_types = []
        
        for surf_gauss in scene.get_surface_gaussians():
            all_gaussians.append((surf_gauss, 'surface'))
        
        for vol_gauss in scene.get_volume_gaussians():
            all_gaussians.append((vol_gauss, 'volume'))
        
        if len(all_gaussians) == 0:
            return final_image
        
        all_depths = []
        all_xy = []
        all_info = []
        
        for gaussians, gtype in all_gaussians:
            xy, depth = camera.project_points(gaussians.get_xyz)
            for i in range(len(depth)):
                all_depths.append(depth[i].item())
                all_xy.append(xy[i])
                all_info.append((gaussians, gtype, i))
        
        sorted_indices = sorted(range(len(all_depths)), key=lambda i: all_depths[i])
        
        ys = torch.arange(H, device=device)
        xs = torch.arange(W, device=device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        
        for idx in sorted_indices:
            gaussians, gtype, g_idx = all_info[idx]
            xy_pos = all_xy[idx]
            
            if gtype == 'surface':
                colors = gaussians.get_features[g_idx, 0, :]
                opacity = gaussians.get_opacity[g_idx].item()
                scaling = gaussians.get_scaling[g_idx]
                radius = scaling.mean() * 100.0
                
                x0, y0 = xy_pos
                
                if radius < 1e-3:
                    continue
                
                xmin = int(max(0, x0 - radius))
                xmax = int(min(W - 1, x0 + radius))
                ymin = int(max(0, y0 - radius))
                ymax = int(min(H - 1, y0 + radius))
                
                if xmin >= xmax or ymin >= ymax:
                    continue
                
                dx = grid_x[ymin:ymax, xmin:xmax] - x0
                dy = grid_y[ymin:ymax, xmin:xmax] - y0
                
                weight = torch.exp(-0.5 * ((dx / radius) ** 2 + (dy / radius) ** 2))
                alpha_surf = opacity * weight
                
                contrib = alpha_surf * transmittance[ymin:ymax, xmin:xmax]
                
                final_image[:, ymin:ymax, xmin:xmax] += (
                    colors.view(3, 1, 1) * contrib
                )
                transmittance[ymin:ymax, xmin:xmax] *= (1.0 - alpha_surf)
                
            else:
                density = gaussians.get_density[g_idx].item()
                phase_color = gaussians.get_phase_color[g_idx]
                opacity = gaussians.get_opacity[g_idx].item()
                scaling = gaussians.get_scaling[g_idx]
                radius = scaling.mean() * 100.0
                
                x0, y0 = xy_pos
                
                if radius < 1e-3:
                    continue
                
                xmin = int(max(0, x0 - radius))
                xmax = int(min(W - 1, x0 + radius))
                ymin = int(max(0, y0 - radius))
                ymax = int(min(H - 1, y0 + radius))
                
                if xmin >= xmax or ymin >= ymax:
                    continue
                
                dx = grid_x[ymin:ymax, xmin:xmax] - x0
                dy = grid_y[ymin:ymax, xmin:xmax] - y0
                
                weight = torch.exp(-0.5 * ((dx / radius) ** 2 + (dy / radius) ** 2))
                alpha_vol = density * weight * opacity
                alpha_vol = alpha_vol.clamp(0, 1)
                
                contrib = alpha_vol * transmittance[ymin:ymax, xmin:xmax]
                
                final_image[:, ymin:ymax, xmin:xmax] += (
                    phase_color.view(3, 1, 1) * contrib
                )
                transmittance[ymin:ymax, xmin:xmax] *= (1.0 - alpha_vol)
        
        return final_image.clamp(0, 1)


def composite_surface_and_volume(scene, camera, bg_color):
    compositor = SurfaceVolumeCompositor()
    return compositor.composite_scene(scene, camera, bg_color)