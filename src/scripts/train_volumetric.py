import argparse
import os
import sys
import csv
from typing import NamedTuple
import torch
import numpy as np
from tqdm import tqdm

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from src.datasets.blender import BlenderDataset
from src.datasets.colmap import ColmapDataset
from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet
from src.rendering.rasterizer import render_gaussians
from src.training.losses import PhotometricLoss

# ----------------------------
# Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Train Volumetric Gaussian Splatting")
parser.add_argument("--data_path", required=True, type=str, help="Path to dataset")
parser.add_argument("--output_path", type=str, default=None, help="Custom output directory")
parser.add_argument("--iterations", dest="num_iter", type=int, default=30_000)
parser.add_argument("--save_interval", type=int, default=7000)
parser.add_argument("--init_points", type=int, default=5000)
parser.add_argument("--fog_reg", type=float, default=0.0001, help="Fog density regularization")
parser.add_argument("--fog_grid_size", type=int, default=32, help="Initial voxel grid size")
parser.add_argument("--simulate_fog", action="store_true", help="Blend training images with white to force fog learning")
parser.add_argument("--lock_surface", action="store_true", help="Freeze surface gaussians")
parser.add_argument("--disable_fog", action="store_true", help="Train only surface")
parser.add_argument("--pretrained_surface", type=str, default=None, help="Path to pretrained surface checkpoint")
args = parser.parse_args()

# ----------------------------
# Constants & Config
# ----------------------------
DATA_PATH = args.data_path
NUM_ITERS = args.num_iter
CHECKPOINT_PATH = args.output_path or f"outputs/checkpoints/{os.path.basename(os.path.normpath(DATA_PATH))}_volumetric"

class TrainingArgs(NamedTuple):
    percent_dense: float = 0.01
    position_lr_init: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    densify_from_iter: int = 500
    densify_until_iter: int = 15_000
    densification_interval: int = 100
    opacity_reset_interval: int = 3000
    densify_grad_threshold: float = 0.0002
    min_opacity: float = 0.005

def image_to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    with open(os.path.join(CHECKPOINT_PATH, "config.txt"), "w") as f:
        for k, v in vars(args).items(): f.write(f"{k}: {v}\n")

    # 1. Load Dataset
    print("Loading dataset...")
    if os.path.exists(os.path.join(DATA_PATH, "transforms_train.json")) or os.path.exists(os.path.join(DATA_PATH, "transforms.json")):
        dataset = BlenderDataset(DATA_PATH)
        camera_convention = "opengl"
    elif os.path.exists(os.path.join(DATA_PATH, "sparse", "0")):
        dataset = ColmapDataset(DATA_PATH)
        camera_convention = "opencv"
    else:
        raise ValueError("Unknown dataset format")

    # 2. Compute Scene Bounds (Adaptive)
    print("Initializing Surface Gaussians...")
    pcd = dataset.get_point_cloud(args.init_points)
    pcd_xyz = torch.tensor(pcd.points).float()
    pcd_min, pcd_max = pcd_xyz.min(dim=0)[0], pcd_xyz.max(dim=0)[0]
    scene_center = (pcd_min + pcd_max) / 2.0
    scene_radius = (pcd_max - pcd_min).max() / 2.0
    print(f"Scene Radius: {scene_radius:.4f}")

    # 3. Setup Surface Model
    gaussians = GaussianSet(sh_degree=0)
    if args.pretrained_surface and os.path.exists(args.pretrained_surface):
        print(f"Loading surface from {args.pretrained_surface}...")
        state = torch.load(args.pretrained_surface, map_location=device)
        try:
            gaussians.load_state_dict(state)
        except RuntimeError:
             # Fallback for manual parameter assignment
             for k in ["_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"]:
                 if k in state: setattr(gaussians, k, torch.nn.Parameter(state[k]))
        gaussians.to(device)
    else:
        gaussians.create_from_pcd(pcd, spatial_lr_scale=float(scene_radius))
    
    opt = TrainingArgs()
    if args.lock_surface:
        print("Freezing Surface Model.")
        gaussians.training_setup(opt)
        gaussians.optimizer = None 
    else:
        gaussians.training_setup(opt)

    # 4. Setup Fog Model
    fog_gaussians = None
    if not args.disable_fog:
        print("Initializing Volumetric Fog...")
        fog_gaussians = VolumetricGaussianSet(sh_degree=0)
        fog_gaussians.create_from_grid(
            center=scene_center.to(device),
            radius=scene_radius.item() * 1.5, 
            grid_size=args.fog_grid_size
        )
        # Use higher learning rates for fluid structure
        fog_args = TrainingArgs(position_lr_init=0.0016, opacity_lr=0.05, feature_lr=0.005)
        fog_gaussians.training_setup(fog_args)

    loss_fn = PhotometricLoss(0.2)
    log_file = os.path.join(CHECKPOINT_PATH, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        csv.writer(f).writerow(["Iteration", "Loss", "SurfacePoints"])

    # 5. Training Loop
    print("Starting training...")
    bar = tqdm(range(1, NUM_ITERS + 1))
    
    for it in bar:
        cam = dataset[torch.randint(0, len(dataset), (1,)).item()]
        
        # Prepare Target
        gt_image = image_to_tensor(cam.image).to(device)
        gt_rgb, gt_alpha = gt_image[:3], gt_image[3:4]
        
        # Random background
        bg_color = torch.rand(3, device=device) if torch.rand(1) < 0.3 else (torch.zeros(3, device=device) if torch.rand(1) < 0.6 else torch.ones(3, device=device))
        target = gt_rgb * gt_alpha + bg_color[:, None, None] * (1.0 - gt_alpha)

        # Fog Simulation (Insertion Mode)
        if args.simulate_fog:
            fog_intensity = 0.3
            fog_c = torch.ones_like(target) * 0.8 # Light gray
            target = target * (1.0 - fog_intensity) + fog_c * fog_intensity

        if gaussians.optimizer: gaussians.optimizer.zero_grad()
        if fog_gaussians: fog_gaussians.optimizer.zero_grad()

        # Combine Models
        if fog_gaussians:
            combined_xyz = torch.cat([gaussians.get_xyz, fog_gaussians.get_xyz], dim=0)
            combined_shs = torch.cat([gaussians.get_features, fog_gaussians.get_features], dim=0)
            combined_opacity = torch.cat([gaussians.get_opacity, fog_gaussians.get_opacity], dim=0)
            combined_scaling = torch.cat([gaussians.get_scaling, fog_gaussians.get_scaling], dim=0)
            combined_rotation = torch.cat([gaussians.get_rotation, fog_gaussians.get_rotation], dim=0)
        else:
            combined_xyz, combined_shs, combined_opacity, combined_scaling, combined_rotation = \
                gaussians.get_xyz, gaussians.get_features, gaussians.get_opacity, gaussians.get_scaling, gaussians.get_rotation

        # Render
        rendered, viewspace_point_tensor = render_gaussians(
            gaussians=gaussians, camera=cam, bg_color=bg_color, convention=camera_convention,
            means3D=combined_xyz, shs=combined_shs, opacities=combined_opacity, scales=combined_scaling, rotations=combined_rotation
        )

        # Loss
        loss = loss_fn(rendered, target).to(device)
        if fog_gaussians:
            loss += args.fog_reg * torch.abs(fog_gaussians.get_opacity).mean()
        
        loss.backward()

        # Optimization & Densification
        with torch.no_grad():
            n_surface = gaussians.get_xyz.shape[0]
            
            # 1. Collect Stats
            if it < opt.densify_until_iter:
                if not args.lock_surface:
                    grid = viewspace_point_tensor.grad[:n_surface]
                    gaussians.add_densification_stats(grid, grid.norm(dim=-1) > 0)
                if fog_gaussians:
                    f_grid = viewspace_point_tensor.grad[n_surface:]
                    fog_gaussians.add_densification_stats(f_grid, f_grid.norm(dim=-1) > 0)

                # 2. Densify/Prune
                if it > opt.densify_from_iter and it % opt.densification_interval == 0:
                    if not args.lock_surface:
                        gaussians.densify_and_prune(opt.densify_grad_threshold, opt.min_opacity, scene_radius*4.0, 20)
                    if fog_gaussians:
                        # Higher threshold for fog to prevent noise
                        fog_gaussians.densify_and_prune(opt.densify_grad_threshold * 1.5, 0.001, scene_radius*4.0, None)
                
                # 3. Opacity Reset
                if it <= opt.densify_until_iter and it % opt.opacity_reset_interval == 0:
                    if not args.lock_surface: gaussians.reset_opacity()

            if gaussians.optimizer: gaussians.optimizer.step()
            if fog_gaussians: fog_gaussians.optimizer.step()

        # Logging & Saving
        if it % 10 == 0:
            bar.set_description(f"Iter: {it}, Loss: {loss.item():.4f}, Surf: {n_surface}")
            with open(log_file, "a", newline="") as f:
                csv.writer(f).writerow([it, loss.item(), n_surface])

        if it % args.save_interval == 0:
            if not args.lock_surface: torch.save(gaussians.state_dict(), f"{CHECKPOINT_PATH}/surface_{it:05d}.pth")
            if fog_gaussians: torch.save(fog_gaussians.state_dict(), f"{CHECKPOINT_PATH}/fog_{it:05d}.pth")

    print("Saving final checkpoints...")
    torch.save(gaussians.state_dict(), f"{CHECKPOINT_PATH}/surface_final.pth")
    if fog_gaussians: torch.save(fog_gaussians.state_dict(), f"{CHECKPOINT_PATH}/fog_final.pth")

if __name__ == "__main__":
    train()
