import argparse
import os
import sys
from pathlib import Path

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(project_root)

import numpy as np
import torch
from PIL import Image

from src.datasets.blender import BlenderDataset
from src.rendering.rasterizer import render_gaussians
from src.datasets.colmap import ColmapDataset
from src.models.gaussians import GaussianSet


def tensor_to_image(tensor):
    """
    CHW float → HWC uint8
    """
    tensor = tensor.clamp(0.0, 1.0)
    img = (tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str)
    parser.add_argument("--checkpoints", required=True, type=str)
    parser.add_argument("--camera", required=False, type=int, default=0)
    parser.add_argument("--out", required=False, type=str, default='outputs/renders/animation.gif')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if os.path.exists(
        os.path.join(args.data_path, "transforms_train.json")
    ) or os.path.exists(os.path.join(args.data_path, "transforms.json")):
        dataset = BlenderDataset(args.data_path)
        camera_convention = "opengl"
    elif os.path.exists(os.path.join(args.data_path, "sparse", "0")):
        dataset = ColmapDataset(args.data_path)
        camera_convention = "opencv"
    else:
        raise ValueError(
            f"No valid dataset found in {args.data_path}. Expected transforms.json or sparse/0/."
        )

    frames = []
    gaussians = GaussianSet()
    checkpoints = sorted(Path(args.checkpoints).glob("*.pth"), key=os.path.getmtime)
    # Filter checkpoints to only include surface_* or gaussians_* 
    # and exclude fog_* as they have a different structure (density/phase_color)
    checkpoints = [cp for cp in checkpoints if "surface" in cp.name or "gaussians" in cp.name and "fog" not in cp.name]
    
    print('Loading checkpoints ...')
    for cp in checkpoints:
        state = torch.load(cp, map_location=device)
        gaussians.load_checkpoint(state)
        gaussians = gaussians.to(device)
        gaussians.eval()

        cam = dataset[args.camera]
        bg_color = torch.zeros(3, device=device)
        with torch.no_grad():
            image, _ = render_gaussians(
                gaussians=gaussians,
                camera=cam,
                bg_color=bg_color,
                convention=camera_convention,
            )

        img = tensor_to_image(image)
        frames.append(Image.fromarray(img))

    print('Generating gif ...')
    frames[0].save(
        f"{args.out}",
        save_all=True,
        append_images=frames[1:],
        duration=100,
        loop=0
    )

if __name__ == "__main__":
    main()
