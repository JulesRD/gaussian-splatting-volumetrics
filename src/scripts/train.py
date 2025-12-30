import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.datasets.blender import BlenderDataset
from src.models.gaussians import GaussianSet
from src.rendering.rasterizer import render_gaussians


# ----------------------------
# Configuration
# ----------------------------

DATA_PATH = "data/forest"
OUTPUT_DIR = "outputs"
NUM_POINTS = 10_000       # start SMALL
NUM_ITERS = 2_000
LR = 1e-3
LOG_INTERVAL = 10
SAVE_INTERVAL = 50
IMAGE_SCALE = 1.0


# ----------------------------
# Utilities
# ----------------------------

def image_to_tensor(img):
    return torch.from_numpy(img).float().permute(2, 0, 1) / 255.0


def setup_dirs():
    os.makedirs(f"{OUTPUT_DIR}/renders", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/checkpoints", exist_ok=True)


# ----------------------------
# Training
# ----------------------------

def train():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    setup_dirs()

    print("Loading dataset...")
    dataset = BlenderDataset(DATA_PATH)

    print("Initializing Gaussians...")
    pcd = dataset.get_point_cloud(NUM_POINTS)

    gaussians = GaussianSet(sh_degree=0)
    gaussians.create_from_pcd(pcd)
    gaussians = gaussians.to(device)

    optimizer = torch.optim.Adam(gaussians.parameters(), lr=LR)

    bg_color = torch.zeros(3, device=device)

    print("Starting training...")
    for it in range(1, NUM_ITERS + 1):
        cam = dataset[torch.randint(0, len(dataset), (1,)).item()]


        target = image_to_tensor(cam.image).to(device)

        optimizer.zero_grad()

        rendered = render_gaussians(
            gaussians=gaussians,
            camera=cam,
            bg_color=bg_color
        )

        loss = F.mse_loss(rendered, target)
        loss.backward()
        optimizer.step()

        if it % LOG_INTERVAL == 0:
            print(f"[Iter {it:05d}] Loss = {loss.item():.6f}")

        if it % SAVE_INTERVAL == 0:
            torch.save(
                gaussians.state_dict(),
                f"{OUTPUT_DIR}/checkpoints/gaussians_{it:05d}.pth"
            )

    torch.save(
        gaussians.state_dict(),
        f"{OUTPUT_DIR}/checkpoints/gaussians_final.pth"
    )


# ----------------------------
# Entry Point
# ----------------------------

def main():
    torch.manual_seed(0)
    train()


if __name__ == "__main__":
    main()