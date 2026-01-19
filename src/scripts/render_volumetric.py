import argparse
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.datasets.blender import BlenderDataset
from src.datasets.colmap import ColmapDataset
from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet
from src.rendering.rasterizer import render_gaussians

def load_dataset(data_path):
    if os.path.exists(os.path.join(data_path, "transforms_train.json")) or os.path.exists(os.path.join(data_path, "transforms.json")):
        return BlenderDataset(data_path), "opengl"
    elif os.path.exists(os.path.join(data_path, "sparse", "0")):
        return ColmapDataset(data_path), "opencv"
    else:
        raise ValueError(f"No valid dataset found in {data_path}")

def render_scene(surface_path, fog_path, dataset, camera_idx, device, fog_opacity_scale=1.0):
    # 1. Load Surface Model
    surface = GaussianSet(sh_degree=0)
    state = torch.load(surface_path, map_location=device)
    try:
        surface.load_state_dict(state)
    except RuntimeError:
         # Fallback manual load
         for k in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
             if k in state: setattr(surface, k, torch.nn.Parameter(state[k]))
    surface.to(device)
    
    # 2. Load Fog Model (if exists)
    if fog_path and os.path.exists(fog_path):
        fog = VolumetricGaussianSet(sh_degree=0)
        fog_state = torch.load(fog_path, map_location=device)
        try:
            fog.load_state_dict(fog_state)
        except RuntimeError:
            fog._xyz = torch.nn.Parameter(fog_state["_xyz"])
            fog._density = torch.nn.Parameter(fog_state["_density"])
            fog._phase_color = torch.nn.Parameter(fog_state["_phase_color"])
            fog._scaling = torch.nn.Parameter(fog_state["_scaling"])
            fog._rotation = torch.nn.Parameter(fog_state["_rotation"])
        fog.to(device)
        
        # Merge Geometry
        combined_xyz = torch.cat([surface.get_xyz, fog.get_xyz], dim=0)
        combined_scaling = torch.cat([surface.get_scaling, fog.get_scaling], dim=0)
        combined_rotation = torch.cat([surface.get_rotation, fog.get_rotation], dim=0)

        # Merge Color/Features (Handle SH mismatch)
        s_feat, f_feat = surface.get_features, fog.get_features
        if s_feat.shape[1] != f_feat.shape[1]:
            # Pad smaller SH with zeros
            max_sh = max(s_feat.shape[1], f_feat.shape[1])
            if s_feat.shape[1] < max_sh:
                s_feat = torch.cat([s_feat, torch.zeros((s_feat.shape[0], max_sh - s_feat.shape[1], 3), device=device)], dim=1)
            elif f_feat.shape[1] < max_sh:
                f_feat = torch.cat([f_feat, torch.zeros((f_feat.shape[0], max_sh - f_feat.shape[1], 3), device=device)], dim=1)
        
        # Check Transpose safety
        if f_feat.shape[-1] != 3 and f_feat.shape[-2] == 3: f_feat = f_feat.transpose(1, 2)
        combined_shs = torch.cat([s_feat, f_feat], dim=0)

        # Merge Opacity (Handle scaling)
        f_opac = fog.get_opacity
        if fog_opacity_scale != 1.0:
            if camera_idx == 0: print(f"Scaling fog opacity by {fog_opacity_scale}x")
            f_opac = torch.clamp(f_opac * fog_opacity_scale, 0.0, 1.0)
        
        combined_opacity = torch.cat([surface.get_opacity, f_opac], dim=0)

    else:
        # Surface Only
        combined_xyz, combined_shs, combined_opacity, combined_scaling, combined_rotation = \
            surface.get_xyz, surface.get_features, surface.get_opacity, surface.get_scaling, surface.get_rotation

    # 3. Render
    cam = dataset[camera_idx]
    with torch.no_grad():
        image, _ = render_gaussians(
            gaussians=surface, camera=cam, bg_color=torch.zeros(3, device=device),
            convention="opengl" if isinstance(dataset, BlenderDataset) else "opencv",
            means3D=combined_xyz, shs=combined_shs, opacities=combined_opacity,
            scales=combined_scaling, rotations=combined_rotation
        )
    
    return (image.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--surface_ckpt", required=True)
    parser.add_argument("--fog_ckpt", required=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["image", "gif", "compare"], default="image")
    parser.add_argument("--fog_opacity_scale", type=float, default=1.0)
    parser.add_argument("--compare_scales", type=float, nargs="+", default=None)
    parser.add_argument("--duration_ms", type=int, default=100)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset, _ = load_dataset(args.data_path)
    
    if args.mode == "image":
        img = render_scene(args.surface_ckpt, args.fog_ckpt, dataset, 0, device, args.fog_opacity_scale)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        Image.fromarray(img).save(args.output)
        print(f"Saved to {args.output}")

    elif args.mode == "gif":
        frames = []
        for i in range(min(60, len(dataset))):
            print(f"Rendering frame {i}...", end='\r')
            frames.append(Image.fromarray(render_scene(args.surface_ckpt, args.fog_ckpt, dataset, i, device, args.fog_opacity_scale)))
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=args.duration_ms, loop=0)
        print(f"\nSaved GIF to {args.output}")

    elif args.mode == "compare":
        frames = []
        views = [("Clean", None, 1.0)]
        if args.compare_scales:
            for s in args.compare_scales: views.append((f"Fog {s}x", args.fog_ckpt, s))
        else:
            views.append(("Fog 1.0x", args.fog_ckpt, 1.0))
            if abs(args.fog_opacity_scale - 1.0) > 0.1: views.append((f"Fog {args.fog_opacity_scale}x", args.fog_ckpt, args.fog_opacity_scale))

        for i in range(min(60, len(dataset))):
            print(f"Rendering comparison {i}...", end='\r')
            row = []
            for label, ckpt, scale in views:
                img = Image.fromarray(render_scene(args.surface_ckpt, ckpt, dataset, i, device, scale))
                d = ImageDraw.Draw(img)
                d.text((10, 10), label, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
                row.append(np.array(img))
            frames.append(Image.fromarray(np.concatenate(row, axis=1)))
        
        frames[0].save(args.output, save_all=True, append_images=frames[1:], duration=args.duration_ms, loop=0)
        print(f"\nSaved Comparison to {args.output}")

if __name__ == "__main__":
    main()