import torch
import math
import numpy as np
from pathlib import Path
from PIL import Image
from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet
from src.models.scene import Scene
from src.rendering.camera import Camera
from src.rendering.composition import composite_surface_and_volume

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

print("=== Création des 3 cubes ===\n")

surf_points = []
surf_colors = []

print("1. Cube VERT (gauche, avec fog)...")
for x in np.linspace(-0.3, 0.3, 5):
    for y in np.linspace(-0.3, 0.3, 5):
        for z in np.linspace(-0.3, 0.3, 5):
            surf_points.append([x - 1.5, y, z])
            surf_colors.append([0.2, 1.0, 0.2])
green_count = len(surf_points)
print(f"   {green_count} points à position (-1.5, 0, 0)")

print("\n2. Cube BLEU (centre)...")
for x in np.linspace(-0.3, 0.3, 5):
    for y in np.linspace(-0.3, 0.3, 5):
        for z in np.linspace(-0.3, 0.3, 5):
            surf_points.append([x, y, z])
            surf_colors.append([0.2, 0.4, 1.0])
blue_count = len(surf_points) - green_count
print(f"   {blue_count} points à position (0, 0, 0)")

print("\n3. Cube ROUGE (droite)...")
for x in np.linspace(-0.3, 0.3, 5):
    for y in np.linspace(-0.3, 0.3, 5):
        for z in np.linspace(-0.3, 0.3, 5):
            surf_points.append([x + 1.5, y, z])
            surf_colors.append([1.0, 0.2, 0.2])
red_count = len(surf_points) - green_count - blue_count
print(f"   {red_count} points à position (1.5, 0, 0)")

surf_points = np.array(surf_points)
surf_colors = np.array(surf_colors)

pcd_surf = SyntheticPCD(surf_points, surf_colors)
surface_gaussians = GaussianSet(sh_degree=0)
surface_gaussians.create_from_pcd(pcd_surf)
surface_gaussians._opacity.data.fill_(10.0)
surface_gaussians._scaling.data.fill_(0.25)

print("\n4. Fog BLANC autour du cube vert...")
vol_points = []
vol_colors = []

for i in range(120):
    theta = np.random.uniform(0, 2*np.pi)
    phi = np.random.uniform(0, np.pi)
    r = np.random.uniform(0.5, 1.0)
    
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)
    
    vol_points.append([x - 1.5, y, z])
    vol_colors.append([0.9, 0.95, 1.0])

print(f"   {len(vol_points)} points autour du cube vert")

vol_points = np.array(vol_points)
vol_colors = np.array(vol_colors)

pcd_vol = SyntheticPCD(vol_points, vol_colors)
fog_gaussians = VolumetricGaussianSet(sh_degree=0)
fog_gaussians.create_from_pcd(pcd_vol, fog_density=0.3, fog_color=(0.9, 0.95, 1.0))
fog_gaussians._opacity.data.fill_(4.0)
fog_gaussians._scaling.data.fill_(0.4)

scene = Scene().to(device)
scene.add_surface_gaussians(surface_gaussians)
scene.add_volume_gaussians(fog_gaussians)

print(f"\n=== Scène créée ===")
print(f"Surface gaussians: {scene.count_surface_gaussians()}")
print(f"Volume gaussians: {scene.count_volume_gaussians()}")

output_path = Path("output")
output_path.mkdir(exist_ok=True)

camera = Camera(512, 512, math.radians(60)).to(device)
bg_color = torch.tensor([0.02, 0.02, 0.05], device=device)

frames = []
num_frames = 60

print(f"\n=== Rendu de {num_frames} frames (rotation 360°) ===")

for frame in range(num_frames):
    angle = (frame / num_frames) * 2 * np.pi
    radius = 4.5
    
    eye_x = radius * np.cos(angle)
    eye_z = radius * np.sin(angle)
    eye_y = 2.0
    
    R, T = look_at(
        eye=(eye_x, eye_y, eye_z),
        target=(0, 0, 0),
        up=(0, 1, 0)
    )
    camera.set_pose(R.to(device), T.to(device))
    
    image = composite_surface_and_volume(scene, camera, bg_color)
    
    image_np = image.detach().cpu().numpy()
    image_np = np.clip(image_np * 255, 0, 255).astype(np.uint8)
    image_np = np.transpose(image_np, (1, 2, 0))
    
    frames.append(image_np)
    
    if frame == 0:
        img = Image.fromarray(image_np)
        img.save(output_path / "three_cubes_frame00.png")
        print(f"  Frame 0 (vue de face)")
    elif frame == 15:
        img = Image.fromarray(image_np)
        img.save(output_path / "three_cubes_frame15.png")
        print(f"  Frame 15 (vue de côté)")
    elif frame == 30:
        img = Image.fromarray(image_np)
        img.save(output_path / "three_cubes_frame30.png")
        print(f"  Frame 30 (vue de dos)")
    elif frame == 45:
        img = Image.fromarray(image_np)
        img.save(output_path / "three_cubes_frame45.png")
        print(f"  Frame 45 (vue de côté)")
    
    if frame % 10 == 0 and frame not in [0, 30]:
        print(f"  Frame {frame}/{num_frames}")

try:
    import imageio
    video_path = output_path / "three_cubes_rotation.mp4"
    imageio.mimsave(str(video_path), frames, fps=30)
    print(f"\n✓ Vidéo sauvegardée: {video_path}")
except ImportError:
    print("\n⚠ Imageio non disponible pour la vidéo")

print(f"\n✓ Images clés sauvegardées:")
print(f"  - output/three_cubes_frame00.png (face)")
print(f"  - output/three_cubes_frame15.png (côté)")
print(f"  - output/three_cubes_frame30.png (dos)")
print(f"  - output/three_cubes_frame45.png (côté)")

print(f"\nPour visualiser:")
print(f"  xdg-open output/three_cubes_frame00.png")
print(f"  xdg-open output/three_cubes_rotation.mp4")

try:
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    images = [
        ("output/three_cubes_frame00.png", "Vue 0° (Face)", axes[0, 0]),
        ("output/three_cubes_frame15.png", "Vue 90° (Côté)", axes[0, 1]),
        ("output/three_cubes_frame30.png", "Vue 180° (Dos)", axes[1, 0]),
        ("output/three_cubes_frame45.png", "Vue 270° (Côté)", axes[1, 1]),
    ]
    
    for path, title, ax in images:
        img = Image.open(path)
        ax.imshow(img)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
    
    plt.suptitle("3 Cubes - Cube VERT avec Fog", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig("output/three_cubes_views.png", dpi=150, bbox_inches='tight')
    print(f"\n✓ Compilation des vues: output/three_cubes_views.png")
    plt.show()
except:
    print("\n(matplotlib non disponible pour la compilation)")

print(f"\n=== Description de la scène ===")
print(f"Cube VERT (gauche):  position (-1.5, 0, 0) AVEC fog blanc")
print(f"Cube BLEU (centre):  position (0, 0, 0) SANS fog")
print(f"Cube ROUGE (droite): position (1.5, 0, 0) SANS fog")
print(f"\nLe fog autour du cube vert devrait créer un halo lumineux blanc")
print(f"qui contraste avec les cubes bleu et rouge nets.")
