from typing import Optional, List, Union
import torch
import torch.nn as nn
from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet


class Scene(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.surface_gaussians: List[GaussianSet] = []
        self.volume_gaussians: List[VolumetricGaussianSet] = []
        self.scene_bounds = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else 
                                  ("mps" if torch.backends.mps.is_available() else "cpu"))
    
    def add_surface_gaussians(self, gaussian_set: GaussianSet) -> int:
        if not isinstance(gaussian_set, GaussianSet):
            raise TypeError("Expected GaussianSet instance")
        self.surface_gaussians.append(gaussian_set)
        return len(self.surface_gaussians) - 1
    
    def add_volume_gaussians(self, volumetric_set: VolumetricGaussianSet) -> int:
        if not isinstance(volumetric_set, VolumetricGaussianSet):
            raise TypeError("Expected VolumetricGaussianSet instance")
        self.volume_gaussians.append(volumetric_set)
        return len(self.volume_gaussians) - 1
    
    def get_surface_gaussians(self, index: Optional[int] = None) -> Union[List[GaussianSet], GaussianSet]:
        if index is not None:
            if 0 <= index < len(self.surface_gaussians):
                return self.surface_gaussians[index]
            else:
                raise IndexError(f"Surface gaussian index {index} out of range")
        return self.surface_gaussians
    
    def get_volume_gaussians(self, index: Optional[int] = None) -> Union[List[VolumetricGaussianSet], VolumetricGaussianSet]:
        if index is not None:
            if 0 <= index < len(self.volume_gaussians):
                return self.volume_gaussians[index]
            else:
                raise IndexError(f"Volume gaussian index {index} out of range")
        return self.volume_gaussians
    
    def count_surface_gaussians(self) -> int:
        return sum(gs.get_xyz.shape[0] for gs in self.surface_gaussians)
    
    def count_volume_gaussians(self) -> int:
        return sum(vgs.get_xyz.shape[0] for vgs in self.volume_gaussians)
    
    def count_total_gaussians(self) -> int:
        return self.count_surface_gaussians() + self.count_volume_gaussians()
    
    def clear(self):
        self.surface_gaussians.clear()
        self.volume_gaussians.clear()
    
    def to(self, device: torch.device):
        self.device = device
        for gs in self.surface_gaussians:
            gs.to(device)
        for vgs in self.volume_gaussians:
            vgs.to(device)
        return self
    
    def train(self, mode: bool = True):
        super().train(mode)
        for gs in self.surface_gaussians:
            gs.train(mode)
        for vgs in self.volume_gaussians:
            vgs.train(mode)
        return self
    
    def eval(self):
        return self.train(False)
    
    def parameters(self):
        params = []
        for gs in self.surface_gaussians:
            params.extend(gs.parameters())
        for vgs in self.volume_gaussians:
            params.extend(vgs.parameters())
        return params
    
    

