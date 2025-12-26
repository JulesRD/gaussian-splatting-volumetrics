import torch
import torch.nn as nn
import numpy as np

class GaussianSet(nn.Module):
    def __init__(self, sh_degree=0):
        super().__init__()
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        
        # Parameters (will be nn.Parameter)
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        
        # Optimizer hook
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        
        self.setup_functions()

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.covariance_activation = build_covariance_from_scaling_rotation
        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = lambda x: torch.log(x/(1-x))
        self.rotation_activation = torch.nn.functional.normalize

    def create_from_pcd(self, pcd, spatial_lr_scale=1):
        self.spatial_lr_scale = spatial_lr_scale
        device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
        
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().to(device)
        fused_color = torch.tensor(np.asarray(pcd.colors)).float().to(device) # RGB 0-1
        
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().to(device)
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print(f"Number of points at initialisation: {fused_point_cloud.shape[0]} on {device}")

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().to(device)), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device=device)
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device=device))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

# Placeholder helper functions - normally these would be in utils or CUDA kernels
def build_scaling_rotation(s, r):
    # Placeholder: In a real impl this constructs the 3x3 matrix
    # For now we assume the renderer handles this or we implement it later
    pass

def strip_symmetric(sym):
    return sym

def distCUDA2(points):
    # Simple k-nearest neighbor distance approximation
    # For N points, this can be slow in python, assuming minimal implementation for now
    # returning mean distance to 3 nearest neighbors
    # This is a placeholder for the actual KNN CUDA call
    return torch.ones(points.shape[0], device=points.device) * 0.1