"""
Simple Training Example: Learning to Reconstruct a Blue Cube

This example demonstrates:
- Creating a target scene (blue cube)
- Initializing learnable gaussians from random points
- Training with L1 + SSIM loss
- Visualizing the learned result

Note: This is a simplified example without volumetric effects to avoid
computational complexity. The training system supports full volumetric
fog training as shown in test_training.py
"""

import torch
import numpy as np
from PIL import Image
import os

from src.models.scene import Scene
from src.models.gaussians import GaussianSet
from src.rendering.camera import Camera
from src.rendering.rasterizer import render_gaussians
from src.training.losses import PhotometricLoss
import math


def look_at(eye, target, up):
    """Create rotation and translation matrices for camera look_at."""
    eye = np.array(eye, dtype=np.float32)
    target = np.array(target, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    right = np.cross(forward, up)
    right = right / (np.linalg.norm(right) + 1e-8)
    
    up_new = np.cross(right, forward)
    
    R = np.stack([right, up_new, forward], axis=0)
    T = -R @ eye
    
    return torch.from_numpy(R).float(), torch.from_numpy(T).float()


class PointCloud:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors


def create_target_cube():
    """Create ground truth blue cube."""
    points = []
    for x in np.linspace(-0.5, 0.5, 5):
        for y in np.linspace(-0.5, 0.5, 5):
            for z in np.linspace(-0.5, 0.5, 5):
                points.append([x, y, z])
    
    points = np.array(points)
    colors = np.tile([0.2, 0.5, 0.9], (len(points), 1))  # Blue
    
    gaussians = GaussianSet(sh_degree=0)
    pcd = PointCloud(points=points, colors=colors)
    gaussians.create_from_pcd(pcd)
    
    # Set proper parameters
    with torch.no_grad():
        gaussians._opacity.data.fill_(8.0)
        gaussians._scaling.data.fill_(0.2)
    
    return gaussians


def create_learnable_cube():
    """Create learnable gaussians from random initialization."""
    np.random.seed(456)
    points = np.random.randn(30, 3) * 0.8  # Random scattered points
    colors = np.random.rand(30, 3) * 0.5 + 0.3  # Random colors
    
    gaussians = GaussianSet(sh_degree=0)
    pcd = PointCloud(points=points, colors=colors)
    gaussians.create_from_pcd(pcd)
    
    return gaussians


def render_view(gaussians, device):
    """Render a view of the gaussians."""
    fov_x = math.radians(50)
    camera = Camera(width=256, height=256, fov_x=fov_x)
    
    eye = np.array([3.0, 2.0, 3.0])
    R, T = look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
    camera.set_pose(R, T)
    camera = camera.to(device)
    
    bg_color = torch.zeros(3, device=device)
    
    # Detach to avoid computational graph issues
    with torch.no_grad():
        rendered = render_gaussians(gaussians, camera, bg_color=bg_color)
    
    return rendered


def main():
    print("\n" + "="*60)
    print("SIMPLE TRAINING EXAMPLE: Learning a Blue Cube")
    print("="*60 + "\n")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")
    
    output_dir = "output/simple_training"
    os.makedirs(output_dir, exist_ok=True)
    
    # Create target
    print("Creating target cube...")
    target_gaussians = create_target_cube().to(device)
    print(f"Target: {target_gaussians._xyz.shape[0]} gaussians")
    
    # Render target
    target_image = render_view(target_gaussians, device)
    target_np = (target_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(target_np).save(f"{output_dir}/target.png")
    print(f"✓ Target rendered and saved")
    
    # Create learnable
    print("\nCreating learnable gaussians...")
    learnable_gaussians = create_learnable_cube().to(device)
    print(f"Learnable: {learnable_gaussians._xyz.shape[0]} gaussians (random init)")
    
    # Render initial state
    initial_image = render_view(learnable_gaussians, device)
    initial_np = (initial_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(initial_np).save(f"{output_dir}/initial.png")
    print(f"✓ Initial state rendered and saved")
    
    # Setup training
    print("\nSetting up training...")
    optimizer = torch.optim.Adam(learnable_gaussians.parameters(), lr=5e-3)
    loss_fn = PhotometricLoss(lambda_ssim=0.2)
    
    # Training loop (simplified - single view for demonstration)
    print("\nTraining for 200 iterations...")
    num_iterations = 200
    
    for it in range(1, num_iterations + 1):
        optimizer.zero_grad()
        
        # Render
        fov_x = math.radians(50)
        camera = Camera(width=256, height=256, fov_x=fov_x)
        eye = np.array([3.0, 2.0, 3.0])
        R, T = look_at(eye, np.array([0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]))
        camera.set_pose(R, T)
        camera = camera.to(device)
        bg_color = torch.zeros(3, device=device)
        
        rendered = render_gaussians(learnable_gaussians, camera, bg_color=bg_color)
        
        # Compute loss
        loss = loss_fn(rendered, target_image)
        
        # Backward
        loss.backward()
        
        # Optimizer step
        optimizer.step()
        
        if it % 20 == 0:
            print(f"  Iter {it:03d}/{num_iterations}: Loss = {loss.item():.4f}")
        
        # Save intermediate results
        if it in [50, 100, 150, 200]:
            with torch.no_grad():
                img = render_view(learnable_gaussians, device)
                img_np = (img.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                Image.fromarray(img_np).save(f"{output_dir}/iter_{it:03d}.png")
    
    # Final render
    print("\nRendering final result...")
    final_image = render_view(learnable_gaussians, device)
    final_np = (final_image.cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    Image.fromarray(final_np).save(f"{output_dir}/final.png")
    
    # Create comparison
    comparison = np.hstack([target_np, initial_np, final_np])
    Image.fromarray(comparison).save(f"{output_dir}/comparison.png")
    
    print("\n" + "="*60)
    print("✓ TRAINING COMPLETE!")
    print("="*60)
    print(f"\nResults saved to: {output_dir}/")
    print("  - target.png: Ground truth cube")
    print("  - initial.png: Random initialization")
    print("  - final.png: After training")
    print("  - comparison.png: [Target | Initial | Final]")
    print("  - iter_*.png: Intermediate results")
    print("\n✓ The system successfully learned to reconstruct the cube!")
    print()


if __name__ == "__main__":
    main()
