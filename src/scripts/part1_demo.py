import sys
import os
import torch
import numpy as np
from plyfile import PlyData, PlyElement

# Add project root to path
sys.path.append(os.getcwd())

from src.datasets.blender import BlenderDataset
from src.models.gaussians import GaussianSet

def save_ply(path, xyz, color):
    # Helper to save a .ply file for visualization
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    elements = np.empty(xyz.shape[0], dtype=dtype)
    elements['x'] = xyz[:, 0]
    elements['y'] = xyz[:, 1]
    elements['z'] = xyz[:, 2]
    elements['red'] = (color[:, 0] * 255).astype(np.uint8)
    elements['green'] = (color[:, 1] * 255).astype(np.uint8)
    elements['blue'] = (color[:, 2] * 255).astype(np.uint8)
    
    el = PlyElement.describe(elements, 'vertex')
    PlyData([el]).write(path)
    print(f"-> Saved debug point cloud to {path}")

def main():
    print("--- Part 1: Data & Model Initialization Demo ---")
    
    # 1. Load Dataset
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
    else:
        dataset_path = "dummy_data"
        
    print(f"\n1. Loading Dataset ({dataset_path})...")
    
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found.")
        return

    dataset = BlenderDataset(dataset_path, split="train")
    print(f"   Loaded {len(dataset)} cameras.")
    print(f"   First camera info: Res={dataset[0].width}x{dataset[0].height}, FOV={dataset[0].FovX:.2f}")

    # 2. Generate Initial Point Cloud
    print("\n2. Generating Initialization Point Cloud...")
    pcd = dataset.get_point_cloud()
    print(f"   Generated {pcd.points.shape[0]} random points within camera frustums.")

    # 3. Initialize Gaussian Model
    print("\n3. Initializing Gaussian Model...")
    gaussians = GaussianSet(sh_degree=3)
    gaussians.create_from_pcd(pcd)
    
    # Check internal tensors
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    colors = gaussians.get_features[:, 0, :].detach().cpu().numpy() # Get DC (base color)
    
    print(f"   Gaussian Model Initialized:")
    print(f"   - Positions Shape: {xyz.shape}")
    print(f"   - Colors Shape:    {colors.shape}")
    print(f"   - Opacities Shape: {gaussians.get_opacity.shape}")

    # 4. Save output
    print("\n4. Saving Result...")
    save_ply("initial_state.ply", xyz, colors)
    
    print("\nDone! You can open 'initial_state.ply' in Blender/MeshLab to see the initialized sparse point cloud.")

if __name__ == "__main__":
    main()
