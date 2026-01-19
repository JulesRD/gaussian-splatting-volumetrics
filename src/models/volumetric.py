from src.models.gaussians import GaussianSet
import torch
import torch.nn as nn
import numpy as np


class VolumetricGaussianSet(GaussianSet):
    def __init__(self, sh_degree=0):
        super().__init__(sh_degree)
        
        # Volumetric Params
        self._density = torch.empty(0)
        self._phase_color = torch.empty(0)
        
        # Remove Surface Params
        del self._opacity, self._features_dc, self._features_rest

    @property
    def get_opacity(self):
        return self.opacity_activation(self._density)
    
    @property
    def get_features(self):
        # Convert RGB Phase Color to SH (Degree 0)
        n_points = self._xyz.shape[0]
        sh_dim = (self.max_sh_degree + 1) ** 2
        features = torch.zeros((n_points, sh_dim, 3), device=self._xyz.device)
        
        # RGB -> SH DC
        dc = (self._phase_color - 0.5) / 0.28209479177387814
        features[:, 0, :] = dc
        return features

    def create_from_grid(self, center, radius, grid_size=32, fog_density=0.01, fog_color=(0.8, 0.8, 0.8)):
        device = torch.device("cuda")
        
        # 1. Create Cylinder/Box Grid
        x = torch.linspace(center[0]-radius, center[0]+radius, grid_size, device=device)
        y = torch.linspace(center[1]-radius, center[1]+radius, grid_size, device=device)
        z = torch.linspace(center[2]-radius, center[2]+radius, grid_size, device=device)
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        grid = torch.stack([xx.flatten(), yy.flatten(), zz.flatten()], dim=1)
        n_points = grid.shape[0]
        
        print(f"Initialized Fog Grid: {n_points} voxels")

        # 2. Init Parameters
        self._xyz = nn.Parameter(grid.requires_grad_(True))
        
        # Uniform Scaling
        voxel_size = (2.0 * radius) / grid_size
        scales = torch.log(torch.ones((n_points, 3), device=device) * voxel_size * 1.5)
        self._scaling = nn.Parameter(scales.requires_grad_(True))

        # Identity Rotation
        rots = torch.zeros((n_points, 4), device=device)
        rots[:, 0] = 1
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        
        # Density & Color
        self._density = nn.Parameter(self.inverse_opacity_activation(fog_density * torch.ones((n_points, 1), device=device)).requires_grad_(True))
        self._phase_color = nn.Parameter(torch.tensor(fog_color, device=device).float().repeat(n_points, 1).requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * 1.0, "name": "xyz"},
            {'params': [self._phase_color], 'lr': training_args.feature_lr, "name": "phase_color"},
            {'params': [self._density], 'lr': training_args.opacity_lr, "name": "density"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        if hasattr(viewspace_point_tensor, "grad"): grad = viewspace_point_tensor.grad
        else: grad = viewspace_point_tensor
        
        self.xyz_gradient_accum[update_filter] += torch.norm(grad[update_filter, :2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        # Split Only (No Cloning for Volume)
        self.densify_and_split(grads, max_grad, extent)
        
        # Prune
        prune_mask = (self.get_opacity < min_opacity).squeeze()
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        self.xyz_gradient_accum = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self._xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self._xyz.shape[0]), device="cuda")

    def densification_postfix(self, new_xyz, new_phase_color, new_density, new_scaling, new_rotation):
        d = {"xyz": new_xyz, "phase_color": new_phase_color, "density": new_density, "scaling": new_scaling, "rotation": new_rotation}
        
        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._phase_color = optimizable_tensors["phase_color"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = torch.cat((self.xyz_gradient_accum, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
        self.denom = torch.cat((self.denom, torch.zeros((new_xyz.shape[0], 1), device="cuda")), dim=0)
        self.max_radii2D = torch.cat((self.max_radii2D, torch.zeros((new_xyz.shape[0]), device="cuda")), dim=0)

    def densify_and_split(self, grads, grad_threshold, scene_extent):
        n_init_points = self._xyz.shape[0]
        # Pad gradients
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        
        selected = torch.where(padded_grad >= grad_threshold, True, False)
        # Avoid splitting already tiny voxels
        selected = torch.logical_and(selected, torch.max(self._scaling, dim=1).values > self.percent_dense * scene_extent)

        stds = self._scaling[selected].repeat(2, 1)
        means = torch.zeros((stds.size(0), 3), device="cuda")
        samples = torch.normal(mean=means, std=stds)
        
        # New Params (Split into 2)
        new_xyz = self._xyz[selected].repeat(2, 1) + samples 
        new_scaling = self.scaling_inverse_activation(self._scaling[selected].repeat(2, 1) / 1.6) 
        new_rotation = self._rotation[selected].repeat(2, 1)
        new_phase_color = self._phase_color[selected].repeat(2, 1)
        new_density = self._density[selected].repeat(2, 1)

        self.densification_postfix(new_xyz, new_phase_color, new_density, new_scaling, new_rotation)
        
        # Remove parent points
        prune_filter = torch.cat((selected, torch.zeros(2 * selected.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def prune_points(self, mask):
        valid = ~mask
        optimizable_tensors = self._prune_optimizer(valid)
        self._xyz = optimizable_tensors["xyz"]
        self._phase_color = optimizable_tensors["phase_color"]
        self._density = optimizable_tensors["density"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        
        self.xyz_gradient_accum = self.xyz_gradient_accum[valid]
        self.denom = self.denom[valid]
        self.max_radii2D = self.max_radii2D[valid]
