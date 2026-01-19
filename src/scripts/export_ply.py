import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
from plyfile import PlyData, PlyElement

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet

def save_ply(xyz, f_dc, opacities, scale, rotation, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    xyz = xyz.detach().cpu().numpy()
    normals = np.zeros_like(xyz)
    f_dc = f_dc.detach().contiguous().cpu().numpy()
    if f_dc.ndim == 3: f_dc = f_dc.reshape(f_dc.shape[0], -1) 
    
    opacities = opacities.detach().cpu().numpy()
    scale = scale.detach().cpu().numpy()
    rotation = rotation.detach().cpu().numpy()

    l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
    for i in range(f_dc.shape[1]): l.append('f_dc_{}'.format(i))
    l.append('opacity')
    for i in range(scale.shape[1]): l.append('scale_{}'.format(i))
    for i in range(rotation.shape[1]): l.append('rot_{}'.format(i))

    dtype_full = [(attribute, 'f4') for attribute in l]
    elements = np.empty(xyz.shape[0], dtype=dtype_full)
    attributes = np.concatenate((xyz, normals, f_dc, opacities, scale, rotation), axis=1)
    elements[:] = list(map(tuple, attributes))
    PlyData([PlyElement.describe(elements, 'vertex')]).write(path)

def export_to_ply(checkpoint_path, output_path, fog_checkpoint=None, fog_opacity_scale=1.0, prune_threshold=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Checkpoint: {checkpoint_path}")
    
    # 1. Load Surface
    gaussians = GaussianSet(sh_degree=0)
    state = torch.load(checkpoint_path, map_location=device)
    try:
        gaussians.load_state_dict(state)
    except RuntimeError:
        # Fallback for old/mismatched checkpoints
        if "_density" in state: # Check if actually volume ckpt
            gaussians._xyz = nn.Parameter(state["_xyz"])
            gaussians._scaling = nn.Parameter(state["_scaling"])
            gaussians._rotation = nn.Parameter(state["_rotation"])
            raw_opac = torch.sigmoid(state["_density"]) * fog_opacity_scale
            gaussians._opacity = nn.Parameter(torch.clamp(raw_opac, 0, 1))
            dc = (state["_phase_color"] - 0.5) / 0.2820947
            gaussians._features_dc = nn.Parameter(dc.unsqueeze(1))
        else:
             for k in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
                 if k in state: setattr(gaussians, k, torch.nn.Parameter(state[k]))
    gaussians.to(device)

    # Accumulate Tensors
    xyz_accum = gaussians._xyz
    f_dc_accum = gaussians._features_dc
    op_accum = gaussians.get_opacity
    sc_accum = gaussians._scaling
    rot_accum = gaussians._rotation

    # 2. Load Fog (Optional)
    if fog_checkpoint:
        print(f"Loading Fog: {fog_checkpoint}")
        fog = VolumetricGaussianSet(sh_degree=0)
        f_state = torch.load(fog_checkpoint, map_location=device)
        try:
            fog.load_state_dict(f_state)
        except RuntimeError:
            fog._xyz = nn.Parameter(f_state["_xyz"])
            fog._density = nn.Parameter(f_state["_density"])
            fog._phase_color = nn.Parameter(f_state["_phase_color"])
            fog._scaling = nn.Parameter(f_state["_scaling"])
            fog._rotation = nn.Parameter(f_state["_rotation"])
        fog.to(device)

        # Merge
        xyz_accum = torch.cat([xyz_accum, fog._xyz], dim=0)
        
        # Match SH dimensions
        f_feat = fog.get_features
        if f_dc_accum.shape[1] != f_feat.shape[1]:
            # Simple fix for SH mismatch: pad or slice
             f_feat = f_feat[:, :f_dc_accum.shape[1], :] if f_feat.shape[1] > f_dc_accum.shape[1] else f_feat
        
        f_dc_accum = torch.cat([f_dc_accum, f_feat], dim=0)

        f_opac = fog.get_opacity
        if fog_opacity_scale != 1.0:
            print(f"Scaling Fog Opacity by {fog_opacity_scale}x")
            f_opac = torch.clamp(f_opac * fog_opacity_scale, 0.0, 1.0)
            
        op_accum = torch.cat([op_accum, f_opac], dim=0)
        sc_accum = torch.cat([sc_accum, fog._scaling], dim=0)
        rot_accum = torch.cat([rot_accum, fog._rotation], dim=0)
    
    # 3. Prune
    if prune_threshold > 0.0:
        print(f"Pruning opacity < {prune_threshold}...")
        mask = (op_accum > prune_threshold).squeeze()
        prev = xyz_accum.shape[0]
        xyz_accum, f_dc_accum, op_accum, sc_accum, rot_accum = \
            xyz_accum[mask], f_dc_accum[mask], op_accum[mask], sc_accum[mask], rot_accum[mask]
        print(f"Pruned {prev - xyz_accum.shape[0]} points. Remaining: {xyz_accum.shape[0]}")

    print(f"Exporting {xyz_accum.shape[0]} Gaussians to {output_path}...")
    save_ply(xyz_accum, f_dc_accum, op_accum, sc_accum, rot_accum, output_path)
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--fog_checkpoint', type=str, default=None)
    parser.add_argument('--output', type=str, required=True)
    parser.add_argument('--fog_opacity_scale', type=float, default=1.0)
    parser.add_argument('--prune_threshold', type=float, default=0.0)
    args = parser.parse_args()

    export_to_ply(args.checkpoint, args.output, args.fog_checkpoint, args.fog_opacity_scale, args.prune_threshold)
