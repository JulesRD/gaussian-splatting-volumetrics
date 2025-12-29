from src.models.gaussians import GaussianSet
import torch
import torch.nn as nn


# density is not the same as opacity :

class VolumetricGaussianSet(GaussianSet):
    def __init__(self, sh_degree=0):
        super().__init__(sh_degree)
        
        self._density = torch.empty(0)
        self._phase_color = torch.empty(0)
    
    def create_from_pcd(self, pcd, fog_density=0.05, fog_color=(1.0,1.0,1.0), spatial_lr_scale=1):
        super().create_from_pcd(pcd, spatial_lr_scale)
        
        n_points = self._xyz.shape[0]
        device = self._xyz.device
        
        self._density = nn.Parameter(fog_density * torch.ones((n_points, 1), device=device, dtype=torch.float, requires_grad=True))
        phase_color_array = torch.tensor(fog_color, device=device, dtype=torch.float).repeat(n_points, 1)
        self._phase_color = nn.Parameter(phase_color_array.requires_grad_(True))
    
    @property
    def get_density(self):
        return self._density
    
    @property
    def get_phase_color(self):
        return self._phase_color
    
    def set_density(self, index, value):
        self._density.data[index] = value
    
    def set_phase_color(self, index, color):
        self._phase_color.data[index] = torch.tensor(color, device=self._phase_color.device)
