import torch
import math
import numpy as np
from pathlib import Path
from PIL import Image
from src.models.gaussians import GaussianSet
from src.models.scene import Scene
from src.rendering.camera import Camera
from src.rendering.rasterizer import render_gaussians

def look_at(eye, target, up):
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

class SyntheticPCD:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

points = []
colors = []

print("Création du cube bleu...")
for x in np.linspace(-0.5, 0.5, 8):
    for y in np.linspace(-0.5, 0.5, 8):
        for z in np.linspace(-0.5, 0.5, 8):
            points.append([x, y, z])
            colors.append([0.2, 0.4, 1.0])

print(f"  {len(points)} points pour le cube")

print("Création du rond rouge (face avant, z=0.6)...")
for i in range(15):
    angle = (i / 15) * 2 * np.pi
    radius = 0.15
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = 0.6
    points.append([x, y, z])
    colors.append([1.0, 0.0, 0.0])

print(f"  {15} points pour le rond rouge")

print("Création du rond vert (face arrière, z=-0.6)...")
for i in range(15):
    angle = (i / 15) * 2 * np.pi
    radius = 0.15
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    z = -0.6
    points.append([x, y, z])
    colors.append([0.0, 1.0, 0.0])

print(f"  {15} points pour le rond vert")

points = np.array(points)
colors = np.array(colors)

pcd = SyntheticPCD(points, colors)
gaussians = GaussianSet(sh_degree=0)
gaussians.create_from_pcd(pcd)

gaussians._opacity.data.fill_(10.0)
gaussians._scaling.data.fill_(0.3)

print(f"\nTotal: {len(points)} gaussiennes")

scene = Scene().to(device)
scene.add_surface_gaussians(gaussians)

output_path = Path("output")
output_path.mkdir(exist_ok=True)

camera = Camera(512, 512, math.radians(60)).to(device)
bg_color = torch.zeros(3, device=device)

frames = []

print("\nRendu de 60 frames (rotation 360°)...")
for frame in range(60):
    angle = (frame / 60) * 2 * np.pi
    radius = 3.0
    
    eye_x = radius * np.cos(angle)
    eye_z = radius * np.sin(angle)
    eye_y = 0.0
    
    R, T = look_at(
        eye=(eye_x, eye_y, eye_z),
        target=(0, 0, 0),
        up=(0, 1, 0)
    )
    camera.set_pose(R.to(device), T.to(device))
    
    image = render_gaussians(gaussians, camera, bg_color)
    
    image_np = image.detach().cpu().numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    image_np = np.transpose(image_np, (1, 2, 0))
    
    frames.append(image_np)
    
    if frame == 0:
        print(f"  Frame 0 (face avant avec rond rouge)")
        img = Image.fromarray(image_np)
        img.save(output_path / "vue_avant.png")
    elif frame == 30:
        print(f"  Frame 30 (face arrière avec rond vert)")
        img = Image.fromarray(image_np)
        img.save(output_path / "vue_arriere.png")
    
    if frame % 10 == 0:
        print(f"  Frame {frame}/60")

try:
    import imageio
    video_path = output_path / "cube_rotation.mp4"
    imageio.mimsave(str(video_path), frames, fps=30)
    print(f"\n✓ Vidéo sauvegardée: {video_path}")
except ImportError:
    print("\n⚠ Imageio non disponible pour la vidéo")

print(f"\n✓ Images sauvegardées:")
print(f"  - output/vue_avant.png (face avant avec rond rouge)")
print(f"  - output/vue_arriere.png (face arrière avec rond vert)")
print(f"\nPour visualiser:")
print(f"  xdg-open output/vue_avant.png")
print(f"  xdg-open output/vue_arriere.png")
print(f"  xdg-open output/cube_rotation.mp4")
