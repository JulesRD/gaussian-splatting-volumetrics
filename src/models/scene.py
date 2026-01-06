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
    
    def clear(self):
        self.surface_gaussians.clear()
        self.volume_gaussians.clear()
    
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
    
    


