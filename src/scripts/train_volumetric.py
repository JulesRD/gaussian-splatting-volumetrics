import os
from typing import NamedTuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv

from src.datasets.blender import BlenderDataset
from src.datasets.colmap import ColmapDataset
from src.models.gaussians import GaussianSet
from src.rendering.rasterizer import render_gaussians
from src.training.losses import PhotometricLoss, FogRegularizationLoss

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--num_iter", required=False, type=int)
parser.add_argument("--stop_dense_after", required=False, type=int)
parser.add_argument("--init_points", required=False, type=int)
parser.add_argument("--reset_interval", required=False, type=int)
parser.add_argument("--data_path", required=False, type=str)
parser.add_argument("--dense_th", required=False, type=float)
parser.add_argument("--min_opacity", required=False, type=float)
args = parser.parse_args()

# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = args.data_path if args.data_path else "data/nerf_synthetic/lego"
OUTPUT_DIR = "outputs"
NUM_POINTS = args.init_points if args.init_points else 5_000
NUM_ITERS = args.num_iter if args.num_iter else 15_000
SAVE_INTERVAL = 100

dataset_name = os.path.basename(DATA_PATH)
CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoints/{dataset_name}_volumetric"

# ----------------------------
# Utilities
# ----------------------------

def image_to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1)

def setup_dirs():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)

# ----------------------------
# Training Args
# ----------------------------

class TrainingArgs(NamedTuple):
    percent_dense: float = 0.01
    position_lr_init: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.05
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    densify_from_iter: int = 500
    densify_until_iter: int = args.stop_dense_after if args.stop_dense_after else 10_000
    densification_interval: int = 100
    opacity_reset_interval: int = args.reset_interval if args.reset_interval else 3000
    densify_grad_threshold: float = args.dense_th if args.dense_th else 0.0002
    min_opacity: float = args.min_opacity if args.min_opacity else 0.005

# ----------------------------
# Training Loop
# ----------------------------

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    setup_dirs()

    print("Loading dataset...")
    if os.path.exists(os.path.join(DATA_PATH, "transforms_train.json")) or os.path.exists(os.path.join(DATA_PATH, "transforms.json")):
        dataset = BlenderDataset(DATA_PATH)
        camera_convention = "opengl"
    elif os.path.exists(os.path.join(DATA_PATH, "sparse", "0")):
        dataset = ColmapDataset(DATA_PATH)
        camera_convention = "opencv"
    else:
        raise ValueError(f"No valid dataset found in {DATA_PATH}. Expected transforms.json or sparse/0/.")

    # Calculate scene extent
    cam_centers = []
    for cam in dataset.cameras:
        if cam is not None:
             cam_centers.append(-cam.R.T @ cam.T)
    
    if len(cam_centers) > 0:
        world_centers = np.stack(cam_centers, axis=0)
        scene_center = np.mean(world_centers, axis=0)
        dists = np.linalg.norm(world_centers - scene_center, axis=1)
        scene_radius = np.percentile(dists, 90) * 1.1
    else:
        scene_center = np.zeros(3)
        scene_radius = 1.0

    print(f"Computed Scene Radius: {scene_radius:.4f}")
    adaptive_zfar = max(100.0, scene_radius * 2.0)

    # 1. Surface Gaussians
    print("Initializing Surface Gaussians...")
    pcd = dataset.get_point_cloud(NUM_POINTS)
    gaussians = GaussianSet(sh_degree=0)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=float(scene_radius))
    
    args = TrainingArgs()
    gaussians.training_setup(args)

    # 2. Volume Gaussians (The Fog)
    print("Initializing Volumetric Fog...")
    fog_gaussians = GaussianSet(sh_degree=0)
    # Initialize grid covering the scene
    fog_gaussians.create_from_grid(
        center=scene_center,
        radius=scene_radius * 1.5, # Slightly larger to cover bounds
        grid_size=32,
        fog_density=0.01,
        fog_color=(0.9, 0.9, 0.95)
    )
    
    # Fog Optimizer (Optimizing only opacity/color mostly, static structure)
    fog_args = TrainingArgs(
        position_lr_init=0.000016, # Allow small movement to break grid
        opacity_lr=0.05, 
        feature_lr=0.005,
        scaling_lr=0.001, # Allow slight reshaping
        rotation_lr=0.0
    )
    fog_gaussians.training_setup(fog_args)

    bg_color = torch.zeros(3, device=device)
    loss_fn = PhotometricLoss(0.2)
    
    print("Starting training...")
    
    log_file = os.path.join(CHECKPOINT_PATH, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Iteration", "Loss", "SurfacePoints"])

    bar = tqdm(range(1, NUM_ITERS + 1))
    
    for it in bar:
        cam = dataset[torch.randint(0, len(dataset), (1,)).item()]

        # Load & Composite
        gt_image = image_to_tensor(cam.image).to(device)
        gt_rgb = gt_image[:3, :, :]
        gt_alpha = gt_image[3:4, :, :]

        rand_val = torch.rand(1).item()
        if rand_val < 0.3: bg_color = torch.rand(3, device=device)
        elif rand_val < 0.6: bg_color = torch.zeros(3, device=device)
        else: bg_color = torch.ones(3, device=device)
        
        target = gt_rgb * gt_alpha + bg_color[:, None, None] * (1.0 - gt_alpha)

        gaussians.optimizer.zero_grad()
        fog_gaussians.optimizer.zero_grad()

        # Concatenate Parameters
        combined_xyz = torch.cat([gaussians.get_xyz, fog_gaussians.get_xyz], dim=0)
        combined_shs = torch.cat([gaussians.get_features, fog_gaussians.get_features], dim=0)
        combined_opacity = torch.cat([gaussians.get_opacity, fog_gaussians.get_opacity], dim=0)
        combined_scaling = torch.cat([gaussians.get_scaling, fog_gaussians.get_scaling], dim=0)
        combined_rotation = torch.cat([gaussians.get_rotation, fog_gaussians.get_rotation], dim=0)

        # Render Unified Scene
        rendered, viewspace_point_tensor = render_gaussians(
            gaussians=gaussians, # Pass object for sh_degree reference
            camera=cam,
            bg_color=bg_color,
            convention=camera_convention,
            means3D=combined_xyz,
            shs=combined_shs,
            opacities=combined_opacity,
            scales=combined_scaling,
            rotations=combined_rotation
        )

        # Loss Calculation
        photo_loss = loss_fn(rendered, target).to(device)
        
        # Fog Regularization: Encourage sparsity/low density in fog volume
        fog_reg = 0.001 * torch.abs(fog_gaussians.get_opacity).mean()
        
        loss = photo_loss + fog_reg
        loss.backward()

        with torch.no_grad():
            # Densification (Surface Only)
            n_surface = gaussians.get_xyz.shape[0]
            if it < args.densify_until_iter:
                # Slice the gradient from the full tensor
                surface_grad = viewspace_point_tensor.grad[:n_surface]
                
                gaussians.add_densification_stats(surface_grad, surface_grad.norm(dim=-1) > 0)

                if it > args.densify_from_iter and it % args.densification_interval == 0:
                    size_threshold = 20 if it > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(args.densify_grad_threshold, args.min_opacity, extent=scene_radius*4.0, max_screen_size=size_threshold)
                
                if it <= args.densify_until_iter and (it % args.opacity_reset_interval == 0 or (getattr(dataset, 'white_background', False) and it == args.densify_from_iter)):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            fog_gaussians.optimizer.step()

        if it % SAVE_INTERVAL == 0:
            torch.save(gaussians.state_dict(), f"{CHECKPOINT_PATH}/surface_{it:05d}.pth")
            torch.save(fog_gaussians.state_dict(), f"{CHECKPOINT_PATH}/fog_{it:05d}.pth")
        
        if it % 10 == 0:
            bar.set_description(f"Iter: {it}, Loss: {loss.item():.4f}, SurfPts: {n_surface}")
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([it, loss.item(), n_surface])

    torch.save(gaussians.state_dict(), f"{CHECKPOINT_PATH}/surface_final.pth")
    torch.save(fog_gaussians.state_dict(), f"{CHECKPOINT_PATH}/fog_final.pth")

if __name__ == "__main__":
    torch.manual_seed(0)
    train()
