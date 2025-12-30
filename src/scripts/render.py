import torch
import numpy as np
import open3d as o3d
import os
from PIL import Image

from src.datasets.blender import BlenderDataset
from src. models.gaussians import GaussianSet
from src.rendering.rasterizer import render_gaussians


# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = "data/forest"
CHECKPOINT_PATH = "outputs/checkpoints/gaussians_00250.pth"
OUTPUT_DIR = "outputs/renders"
CAMERA_INDEX = 0


# ----------------------------
# Utilities
# ----------------------------

def tensor_to_image(tensor):
    """
    CHW float → HWC uint8
    """
    tensor = tensor.clamp(0.0, 1.0)
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


# ----------------------------
# Rendering
# ----------------------------

def render_checkpoint():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading dataset...")
    dataset = BlenderDataset(DATA_PATH)

    print("Loading Gaussians...")
    gaussians = GaussianSet(sh_degree=0)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    gaussians.load_checkpoint(state)
    gaussians = gaussians.to(device)
    gaussians.eval()

    cam = dataset[CAMERA_INDEX]

    bg_color = torch.zeros(3, device=device)

    print("Rendering...")
    with torch.no_grad():
        image = render_gaussians(
            gaussians=gaussians,
            camera=cam,
            bg_color=bg_color
        )

    img = tensor_to_image(image)
    out_path = os.path.join(OUTPUT_DIR, f"view_{CAMERA_INDEX:03d}.png")
    Image.fromarray(img).save(out_path)

    print(f"Saved render to {out_path}")

def visualize_gaussians():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    gaussians = GaussianSet(sh_degree=0)
    state = torch.load(CHECKPOINT_PATH, map_location=device)
    gaussians.load_checkpoint(state)
    gaussians = gaussians.to(device)
    gaussians.eval()

    # ----------------------------
    # Extract data
    # ----------------------------
    xyz = gaussians.get_xyz.detach().cpu().numpy()
    colors = gaussians.get_features[:, 0, :].detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy()

    # Optional: opacity-weighted colors
    colors = colors * opacity

    # ----------------------------
    # Open3D point cloud
    # ----------------------------
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(colors.clip(0, 1))

    print(f"Visualizing {xyz.shape[0]} Gaussians")
    o3d.visualization.draw_geometries(
        [pcd],
        window_name="Gaussian Scene (Point Cloud)",
        width=1200,
        height=800,
    )


if __name__ == "__main__":
    #render_checkpoint()
    visualize_gaussians()

