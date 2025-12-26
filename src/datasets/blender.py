import os
import json
import numpy as np
from PIL import Image
from typing import NamedTuple
from torch.utils.data import Dataset

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: float
    FovX: float
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class BlenderDataset(Dataset):
    def __init__(self, data_path, split="train", white_background=False):
        self.data_path = data_path
        self.split = split
        self.white_background = white_background
        
        try:
            with open(os.path.join(self.data_path, f"transforms_{split}.json"), 'r') as f:
                self.meta = json.load(f)
        except FileNotFoundError:
             # Fallback if specific split json doesn't exist, try generic or assume folder structure
             print(f"Warning: transforms_{split}.json not found. Checking for transforms.json")
             with open(os.path.join(self.data_path, "transforms.json"), 'r') as f:
                self.meta = json.load(f)

        self.camera_angle_x = self.meta.get("camera_angle_x", 1.0)
        self.frames = self.meta["frames"]
        
        # Load all cameras
        self.cameras = []
        for idx, frame in enumerate(self.frames):
            cam_info = self.read_camera(idx, frame)
            self.cameras.append(cam_info)

    def read_camera(self, idx, frame):
        filepath = os.path.join(self.data_path, frame["file_path"] + ".png")
        if not os.path.exists(filepath):
             # Try without extension or different extension if needed, but standard is .png
             filepath = os.path.join(self.data_path, frame["file_path"])
        
        # Image Loading
        image = Image.open(filepath)
        im_data = np.array(image.convert("RGBA"))
        
        # Background handling
        bg = np.array([1,1,1]) if self.white_background else np.array([0,0,0])
        
        norm_data = im_data / 255.0
        arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
        image = Image.fromarray(np.array(arr*255.0, dtype=np.uint8), "RGB")
        
        width, height = image.size
        FovX = self.camera_angle_x
        FovY = 2 * np.arctan(np.tan(self.camera_angle_x / 2) * height / width)
        
        # Matrix parsing
        # NeRF 'transform_matrix' is Camera-to-World. 
        # Gaussian Splatting usually expects World-to-Camera (View Matrix).
        c2w = np.array(frame["transform_matrix"])
        
        # Coordinate conversion: OpenGL (NeRF) to OpenCV (Gaussian Splatting internal often uses this)
        # NeRF: +X Right, +Y Up, +Z Back (cam looks -Z)
        # We want to maintain consistency. Let's store C2W and handle inversion in the Camera class later
        # or invert here. Standard 3DGS implementations invert here.
        
        # Inverting C2W to W2C
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed usually for shader usage
        T = w2c[:3, 3]

        return CameraInfo(
            uid=idx,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image=np.array(image),
            image_path=filepath,
            image_name=frame["file_path"],
            width=width,
            height=height
        )

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def get_point_cloud(self):
        # Generate random point cloud inside the bounding box of the cameras
        # This is a fallback if no ply is available
        print("Generating random point cloud...")
        num_pts = 100_000
        # Determine bounding box from cameras
        centers = []
        for cam in self.cameras:
            # T is from w2c, we need c2w center
            # w2c = [R | T] -> c2w = [R.T | -R.T * T]
            # Center is -R.T * T
            R = cam.R.T # Transpose back to normal rotation
            T = cam.T
            center = -R.T @ T
            centers.append(center)
        
        centers = np.array(centers)
        min_bound = np.min(centers, axis=0) - 0.5
        max_bound = np.max(centers, axis=0) + 0.5
        
        xyz = np.random.random((num_pts, 3)) * (max_bound - min_bound) + min_bound
        rgb = np.random.random((num_pts, 3)) # Random colors
        
        return BasicPointCloud(points=xyz, colors=rgb, normals=np.zeros((num_pts, 3)))
