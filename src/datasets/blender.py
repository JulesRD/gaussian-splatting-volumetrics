import os
import json
import numpy as np
from PIL import Image
from typing import NamedTuple
from torch.utils.data import Dataset


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray        # (3, 3) world-to-camera rotation
    T: np.ndarray        # (3,)   world-to-camera translation
    FovY: float
    FovX: float
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int


class BlenderDataset(Dataset):
    def __init__(self, data_path, white_background=False):
        self.data_path = data_path
        self.white_background = white_background

        transforms_path = os.path.join(self.data_path, "transforms.json")
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms.json not found in {data_path}")

        with open(transforms_path, "r") as f:
            self.meta = json.load(f)

        self.camera_angle_x = self.meta["camera_angle_x"]
        self.frames = self.meta["frames"]

        self.cameras = []
        for idx, frame in enumerate(self.frames):
            self.cameras.append(self.read_camera(idx, frame))

    def read_camera(self, idx, frame):
        # ---------- Image loading ----------
        image_path = os.path.join(self.data_path, frame["file_path"])
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = Image.open(image_path)

        # Handle RGB / RGBA safely
        if image.mode == "RGBA":
            rgba = np.array(image).astype(np.float32) / 255.0
            rgb = rgba[..., :3]
            alpha = rgba[..., 3:4]
            bg = np.ones((1, 1, 3)) if self.white_background else np.zeros((1, 1, 3))
            rgb = rgb * alpha + bg * (1.0 - alpha)
            image = (rgb * 255).astype(np.uint8)
        else:
            image = np.array(image.convert("RGB"))

        height, width = image.shape[:2]

        # ---------- FOV ----------
        FovX = self.camera_angle_x
        FovY = 2.0 * np.arctan(
            np.tan(FovX * 0.5) * height / width
        )

        # ---------- Camera matrix ----------
        # NeRF gives camera-to-world
        c2w = np.array(frame["transform_matrix"], dtype=np.float32)

        # Convert to world-to-camera
        w2c = np.linalg.inv(c2w)

        R = w2c[:3, :3]
        T = w2c[:3, 3]

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=image,
            image_path=image_path,
            image_name=os.path.basename(frame["file_path"]),
            width=width,
            height=height,
        )

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def get_point_cloud(self, num_pts=100_000):
        """
        Fallback point cloud initialization.
        Samples points in the bounding box of camera centers.
        """
        print("Generating random point cloud...")

        centers = []
        for cam in self.cameras:
            # Camera center in world coordinates:
            # C = -R^T T
            center = -cam.R.T @ cam.T
            centers.append(center)

        centers = np.stack(centers, axis=0)

        min_bound = centers.min(axis=0) - 0.5
        max_bound = centers.max(axis=0) + 0.5

        xyz = np.random.uniform(min_bound, max_bound, size=(num_pts, 3))
        rgb = np.random.uniform(0.0, 1.0, size=(num_pts, 3))
        normals = np.zeros_like(xyz)

        return BasicPointCloud(
            points=xyz,
            colors=rgb,
            normals=normals
        )


"""
# Example usage
dataset = BlenderDataset("data/forest_other_angle")

cam = dataset[0]
print("Image shape:", cam.image.shape)
print("FOV X (deg):", np.degrees(cam.FovX))
print("Camera center:", -cam.R.T @ cam.T)
"""
