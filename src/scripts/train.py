import argparse
import csv
import os
import sys

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

from typing import NamedTuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.blender import BlenderDataset
from src.datasets.colmap import BasicPointCloud, CameraInfo, ColmapDataset
from src.models.gaussians import GaussianSet
from src.rendering.rasterizer import render_gaussians
from src.training.losses import PhotometricLoss

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
NUM_POINTS = args.init_points if args.init_points else 5_000  # start SMALL
NUM_ITERS = args.num_iter if args.num_iter else 10_000
LR = 1e-3
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
IMAGE_SCALE = 1.0

dataset_name = os.path.basename(DATA_PATH)
CHECKPOINT_PATH = f"{OUTPUT_DIR}/checkpoints/{dataset_name}"


# ----------------------------
# Utilities
# ----------------------------


def image_to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1)


def setup_dirs():
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)


# ----------------------------
# Training
# ----------------------------


class TrainingArgs(NamedTuple):
    percent_dense: float = 0.01
    position_lr_init: float = 0.00016
    feature_lr: float = 0.0025
    opacity_lr: float = 0.025
    scaling_lr: float = 0.005
    rotation_lr: float = 0.001
    densify_from_iter: int = 500
    densify_until_iter: int = args.stop_dense_after if args.stop_dense_after else 15_000
    densification_interval: int = 100
    opacity_reset_interval: int = args.reset_interval if args.reset_interval else 2000
    densify_grad_threshold: float = args.dense_th if args.dense_th else 0.0002
    min_opacity: float = args.min_opacity if args.min_opacity else 0.001


def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    setup_dirs()

    print("Loading dataset...")
    if os.path.exists(
        os.path.join(DATA_PATH, "transforms_train.json")
    ) or os.path.exists(os.path.join(DATA_PATH, "transforms.json")):
        dataset = BlenderDataset(DATA_PATH)
        camera_convention = "opengl"
    elif os.path.exists(os.path.join(DATA_PATH, "sparse", "0")):
        dataset = ColmapDataset(DATA_PATH)
        camera_convention = "opencv"
    else:
        raise ValueError(
            f"No valid dataset found in {DATA_PATH}. Expected transforms.json or sparse/0/."
        )

    print("Initializing Gaussians...")
    pcd = dataset.get_point_cloud(NUM_POINTS)

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
    print(f"Adaptive Zfar: {adaptive_zfar:.4f}")

    gaussians = GaussianSet(sh_degree=0)
    gaussians.create_from_pcd(pcd, spatial_lr_scale=float(scene_radius))

    args = TrainingArgs()
    print(args)
    gaussians.training_setup(args)

    bg_color = torch.zeros(3, device=device)
    loss_fn = PhotometricLoss(0.2)

    print("Starting training...")
    log_file = os.path.join(CHECKPOINT_PATH, "training_log.csv")
    with open(log_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["Iteration", "Loss", "Points", "Avg Opacity", "Avg Scale", "Max Scale"]
        )

    bar = tqdm(range(1, NUM_ITERS + 1))

    for it in bar:
        cam = dataset[torch.randint(0, len(dataset), (1,)).item()]

        # Load RGBA image (already float32 0-1)
        gt_image = image_to_tensor(cam.image).to(device)
        gt_rgb = gt_image[:3, :, :]
        gt_alpha = gt_image[3:4, :, :]

        # Random background color for data augmentation (force geometry learning)
        # Flip between random, black, and white
        rand_val = torch.rand(1).item()
        if rand_val < 0.3:
            bg_color = torch.rand(3, device=device)
        elif rand_val < 0.6:
            bg_color = torch.zeros(3, device=device)
        else:
            bg_color = torch.ones(3, device=device)

        # Composite Ground Truth
        target = gt_rgb * gt_alpha + bg_color[:, None, None] * (1.0 - gt_alpha)

        gaussians.optimizer.zero_grad()

        rendered, viewspace_point_tensor = render_gaussians(
            gaussians=gaussians,
            camera=cam,
            bg_color=bg_color,
            convention=camera_convention,
            zfar=adaptive_zfar,
        )

        loss = loss_fn(rendered, target).to(device)
        loss.backward()

        with torch.no_grad():
            # Densification
            if it < args.densify_until_iter:
                gaussians.add_densification_stats(
                    viewspace_point_tensor, viewspace_point_tensor.grad.norm(dim=-1) > 0
                )

                if (
                    it > args.densify_from_iter
                    and it % args.densification_interval == 0
                ):
                    size_threshold = 20 if it > args.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        args.densify_grad_threshold,
                        args.min_opacity,
                        extent=20.0,
                        max_screen_size=size_threshold,
                    )

                if it <= args.densify_until_iter and (
                    it % args.opacity_reset_interval == 0
                    or (
                        getattr(dataset, "white_background", False)
                        and it == args.densify_from_iter
                    )
                ):
                    gaussians.reset_opacity()

            gaussians.optimizer.step()

        if it % SAVE_INTERVAL == 0:
            torch.save(
                gaussians.state_dict(), f"{CHECKPOINT_PATH}/gaussians_{it:05d}.pth"
            )

        if it % 10 == 0:
            bar.set_description(
                f"Iteration: {it}, Loss: {loss.item():.4f}, Points: {gaussians.get_xyz.shape[0]}"
            )
            avg_opacity = gaussians.get_opacity.mean().item()
            avg_scale = gaussians.get_scaling.mean().item()
            max_scale = gaussians.get_scaling.max().item()
            with open(log_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        it,
                        loss.item(),
                        gaussians.get_xyz.shape[0],
                        avg_opacity,
                        avg_scale,
                        max_scale,
                    ]
                )

    torch.save(
        gaussians.state_dict(),
        f"{CHECKPOINT_PATH}/gaussians_final_{NUM_ITERS}_{gaussians.get_xyz.shape[0]}.pth",
    )


# ----------------------------
# Entry Point
# ----------------------------


def main():
    torch.manual_seed(0)
    train()


if __name__ == "__main__":
    main()
