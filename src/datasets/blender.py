import json
import math
import os
from typing import NamedTuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from rendering import camera


class BasicPointCloud(NamedTuple):
    points: np.ndarray
    colors: np.ndarray
    normals: np.ndarray


class CameraInfo(NamedTuple):
    uid: int
    R: np.ndarray  # (3, 3) world-to-camera rotation
    T: np.ndarray  # (3,)   world-to-camera translation
    FovY: float
    FovX: float
    fx: float
    fy: float
    cx: float
    cy: float
    image: np.ndarray
    image_path: str
    image_name: str
    width: int
    height: int


class BlenderDataset(Dataset):
    def __init__(self, data_path, white_background=False, split="train"):
        self.data_path = data_path
        self.white_background = white_background
        self.split = split

        transforms_path = os.path.join(self.data_path, f"transforms_{split}.json")
        if not os.path.exists(transforms_path):
            raise FileNotFoundError(f"transforms_{split}.json not found in {data_path}")

        with open(transforms_path, "r") as f:
            self.meta = json.load(f)

        self.camera_angle_x = self.meta["camera_angle_x"]
        self.frames = self.meta["frames"]

        self.cameras = []
        for idx, frame in enumerate(self.frames):
            self.cameras.append(self.read_camera(idx, frame))

    def read_camera(self, idx, frame):
        # ---------- Image loading ----------
        clean_path = (
            frame["file_path"][2:]
            if len(frame["file_path"]) > 0 and frame["file_path"][:2] == "./"
            else frame["file_path"]
        )
        if not clean_path.endswith(".png"):
            clean_path += ".png"

        image_path = os.path.join(self.data_path, clean_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        image = Image.open(image_path)

        # Handle RGB / RGBA safely
        if image.mode == "RGBA":
            rgba = np.array(image).astype(np.float32) / 255.0
            image = rgba  # Return RGBA
        else:
            image_rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image = np.concatenate(
                [image_rgb, np.ones_like(image_rgb[..., :1])], axis=-1
            )

        # Override dimensions with actual loaded image size
        height, width = image.shape[:2]

        # ---------- FOV ----------
        FovX = self.camera_angle_x
        FovY = 2.0 * np.arctan(np.tan(FovX * 0.5) * height / width)

        # ---------- Intrinsics ----------
        fx = width / (2 * math.tan(FovX / 2))
        fy = height / (2 * math.tan(FovY / 2))
        cx = width / 2
        cy = height / 2

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
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
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
        Frustum initialization.
        Samples points in front of the cameras, ensuring they are visible.
        This works for both object-centric and forward-facing scenes.
        """
        print(f"Generating point cloud in camera frustums ({num_pts} points)...")

        xyz = []
        rgb = []

        points_per_cam = max(1, num_pts // len(self.cameras))

        for idx, frame in enumerate(self.frames):
            # NeRF/Blender matrix is C2W
            c2w = np.array(frame["transform_matrix"], dtype=np.float32)

            # Extract position and forward vector
            cam_pos = c2w[:3, 3]

            # Col 2 is "Back" in OpenGL (-Z is forward)
            forward = -c2w[:3, 2]

            # Also use Right/Up to spread points laterally
            right = c2w[:3, 0]
            up = c2w[:3, 1]

            # Sample depths (e.g. 0.5 to 6.0 units in front)
            depths = np.random.uniform(0.5, 6.0, size=(points_per_cam, 1))

            # Sample lateral spread (frustum-like)
            # Assuming FOV ~60 deg, tan(30) ~ 0.5
            # Spread proportional to depth
            spread_x = np.random.uniform(-0.5, 0.5, size=(points_per_cam, 1)) * depths
            spread_y = np.random.uniform(-0.5, 0.5, size=(points_per_cam, 1)) * depths

            # P = Pos + Depth * Forward + SpreadX * Right + SpreadY * Up
            pts = cam_pos + depths * forward + spread_x * right + spread_y * up

            xyz.append(pts)
            # Random colors for now
            rgb.append(np.random.uniform(0.0, 1.0, size=(points_per_cam, 3)))

        xyz = np.concatenate(xyz, axis=0)
        rgb = np.concatenate(rgb, axis=0)

        # Subsample if we have too many
        if xyz.shape[0] > num_pts:
            indices = np.random.choice(xyz.shape[0], num_pts, replace=False)
            xyz = xyz[indices]
            rgb = rgb[indices]

        normals = np.zeros_like(xyz)

        return BasicPointCloud(points=xyz, colors=rgb, normals=normals)
