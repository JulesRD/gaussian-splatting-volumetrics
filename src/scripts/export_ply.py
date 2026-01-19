import torch
import numpy as np
import os
from plyfile import PlyData, PlyElement
from src.models.gaussians import GaussianSet
import argparse

# ----------------------------
# Configuration
# ----------------------------

def construct_list_of_attributes(gaussians):
    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    # All channels of SH
    for i in range(gaussians.get_features.shape[1] * gaussians.get_features.shape[2]):
        l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(gaussians.get_scaling.shape[1]):
        l.append('scale_{}'.format(i))
    for i in range(gaussians.get_rotation.shape[1]):
        l.append('rot_{}'.format(i))
    return l

def save_ply(gaussians, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

    xyz = gaussians.get_xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = gaussians._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    f_rest = gaussians._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacities = gaussians._opacity.detach().cpu().numpy()
    scale = gaussians._scaling.detach().cpu().numpy()
    rotation = gaussians._rotation.detach().cpu().numpy()

    dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes(gaussians)]

    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)

def export_to_ply(checkpoint_path, output_path, fog_checkpoint=None):
    print(f"Loading surface checkpoint from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    gaussians = GaussianSet(sh_degree=0) # Make sure this matches training
    state = torch.load(checkpoint_path, map_location=device)
    gaussians.load_checkpoint(state)
    gaussians.to(device)
    
    if fog_checkpoint:
        print(f"Loading fog checkpoint from {fog_checkpoint}...")
        fog = GaussianSet(sh_degree=0)
        fog_state = torch.load(fog_checkpoint, map_location=device)
        fog.load_checkpoint(fog_state)
        fog.to(device)
        
        print("Merging surface and fog...")
        # Concatenate parameters into 'gaussians' object for export
        gaussians._xyz = torch.nn.Parameter(torch.cat([gaussians._xyz, fog._xyz], dim=0))
        gaussians._features_dc = torch.nn.Parameter(torch.cat([gaussians._features_dc, fog._features_dc], dim=0))
        gaussians._features_rest = torch.nn.Parameter(torch.cat([gaussians._features_rest, fog._features_rest], dim=0))
        gaussians._opacity = torch.nn.Parameter(torch.cat([gaussians._opacity, fog._opacity], dim=0))
        gaussians._scaling = torch.nn.Parameter(torch.cat([gaussians._scaling, fog._scaling], dim=0))
        gaussians._rotation = torch.nn.Parameter(torch.cat([gaussians._rotation, fog._rotation], dim=0))
    
    print(f"Exporting {gaussians.get_xyz.shape[0]} Gaussians to {output_path}...")
    save_ply(gaussians, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default="outputs/checkpoints/gaussians_final.pth")
    parser.add_argument('--fog_checkpoint', type=str, default=None, help="Optional path to fog checkpoint")
    parser.add_argument('--output', type=str, default="outputs/point_cloud/iteration_30000/point_cloud.ply")
    args = parser.parse_args()

    export_to_ply(args.checkpoint, args.output, args.fog_checkpoint)
