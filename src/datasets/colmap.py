import collections
import os
import struct
from typing import NamedTuple

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

# ----------------------------
# Data Structures
# ----------------------------


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
    image_name: str
    width: int
    height: int


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"]
)
Camera = collections.namedtuple("Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"]
)
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"]
)

CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
}
CAMERA_MODEL_IDS = dict(
    [(camera_model.model_id, camera_model) for camera_model in CAMERA_MODELS]
)

# ----------------------------
# COLMAP Binary Readers
# ----------------------------


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_cameras_binary(path_to_model_file):
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(fid, 24, "iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = CAMERA_MODEL_IDS[model_id].num_params
            params = read_next_bytes(fid, num_params * 8, "d" * num_params)
            cameras[camera_id] = Camera(
                id=camera_id,
                model=CAMERA_MODEL_IDS[model_id].model_name,
                width=width,
                height=height,
                params=np.array(params),
            )
    return cameras


def read_images_binary(path_to_model_file):
    images = {}
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            binary_image_properties = read_next_bytes(fid, 64, "idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, 8, "Q")[0]
            xys = read_next_bytes(fid, num_points2D * 16, "d" * (num_points2D * 2))
            xys = np.array(xys).reshape((num_points2D, 2))
            point3D_ids = read_next_bytes(fid, num_points2D * 8, "Q" * num_points2D)
            point3D_ids = np.array(point3D_ids)
            images[image_id] = BaseImage(
                id=image_id,
                qvec=qvec,
                tvec=tvec,
                camera_id=camera_id,
                name=image_name,
                xys=xys,
                point3D_ids=point3D_ids,
            )
    return images


def read_points3D_binary(path_to_model_file):
    points3D = {}
    with open(path_to_model_file, "rb") as fid:
        num_points = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_points):
            binary_point_line_properties = read_next_bytes(fid, 43, "QdddBBBd")
            point3D_id = binary_point_line_properties[0]
            xyz = np.array(binary_point_line_properties[1:4])
            rgb = np.array(binary_point_line_properties[4:7])
            error = binary_point_line_properties[7]
            track_length = read_next_bytes(fid, 8, "Q")[0]
            track_elems = read_next_bytes(fid, track_length * 8, "ii" * track_length)
            image_ids = np.array(tuple(map(int, track_elems[0::2])))
            point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
            points3D[point3D_id] = Point3D(
                id=point3D_id,
                xyz=xyz,
                rgb=rgb,
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3D


def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


# ----------------------------
# Dataset Class
# ----------------------------


class ColmapDataset(Dataset):
    def __init__(self, data_path, images_folder="images"):
        self.data_path = data_path
        self.images_folder = images_folder
        self.sparse_path = os.path.join(data_path, "sparse", "0")

        print(f"Loading COLMAP data from {self.sparse_path}...")
        self.cameras_raw = read_cameras_binary(
            os.path.join(self.sparse_path, "cameras.bin")
        )
        self.images_raw = read_images_binary(
            os.path.join(self.sparse_path, "images.bin")
        )
        self.points3D_raw = read_points3D_binary(
            os.path.join(self.sparse_path, "points3D.bin")
        )

        self.cameras = []
        for img_id in sorted(self.images_raw.keys()):
            cam = self.read_camera(img_id)
            if cam is not None:
                self.cameras.append(cam)

        if len(self.cameras) > 0:
            first_cam_id = self.images_raw[list(self.images_raw.keys())[0]].camera_id
            print(
                f"Detected COLMAP Camera Model: {self.cameras_raw[first_cam_id].model}"
            )

    def read_camera(self, img_id):
        img_data = self.images_raw[img_id]
        cam_data = self.cameras_raw[img_data.camera_id]

        # Load Image
        image_path = os.path.join(self.data_path, self.images_folder, img_data.name)
        if not os.path.exists(image_path):
            # Fallback if names are simple 0001.png etc
            print(f"Warning: {image_path} not found.")
            return None

        image = Image.open(image_path)

        # Handle RGB / RGBA
        if image.mode == "RGBA":
            rgba = np.array(image).astype(np.float32) / 255.0
            image = rgba
        else:
            image_rgb = np.array(image.convert("RGB")).astype(np.float32) / 255.0
            image = np.concatenate(
                [image_rgb, np.ones_like(image_rgb[..., :1])], axis=-1
            )

        # Use actual image dimensions to avoid mismatches
        height, width = image.shape[:2]

        scale_x = width / cam_data.width
        scale_y = height / cam_data.height

        # Intrinsics
        cx = width / 2
        cy = height / 2
        if cam_data.model == "PINHOLE" or cam_data.model == "OPENCV":
            fx = cam_data.params[0] * scale_x
            fy = cam_data.params[1] * scale_y
            cx = cam_data.params[2] * scale_x
            cy = cam_data.params[3] * scale_y
        elif cam_data.model == "SIMPLE_PINHOLE":
            fx = cam_data.params[0] * scale_x
            fy = cam_data.params[0] * scale_y
            cx = cam_data.params[1] * scale_x
            cy = cam_data.params[2] * scale_y
        elif cam_data.model == "SIMPLE_RADIAL":
            fx = cam_data.params[0] * scale_x
            fy = cam_data.params[0] * scale_y
            cx = cam_data.params[1] * scale_x
            cy = cam_data.params[2] * scale_y
        elif cam_data.model == "RADIAL":
            fx = cam_data.params[0] * scale_x
            fy = cam_data.params[0] * scale_y
            cx = cam_data.params[1] * scale_x
            cy = cam_data.params[2] * scale_y
        else:
            print(
                f"Warning: Unhandled camera model {cam_data.model}, using width/FOV guess"
            )
            fx = width  # Placeholder
            fy = width

        # Warn if principal point is significantly off-center (may indicate cropping)
        if abs(cx - width / 2) > 1 or abs(cy - height / 2) > 1:
            print(
                f"Warning: Camera {img_id} has off-center principal point (cx={cx:.1f}, cy={cy:.1f}), expected center ({width / 2:.1f}, {height / 2:.1f}). This may affect rendering."
            )

        # Calculate FOV
        FovY = 2 * np.arctan(height / (2 * fy))
        FovX = 2 * np.arctan(width / (2 * fx))

        R = qvec2rotmat(img_data.qvec)
        T = img_data.tvec

        return CameraInfo(
            uid=img_id,
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            image=image,
            image_name=img_data.name,
            width=width,
            height=height,
        )

    def __len__(self):
        return len(self.cameras)

    def __getitem__(self, idx):
        return self.cameras[idx]

    def get_point_cloud(self, num_pts=None):
        """
        Load points from COLMAP binary.
        Ignores num_pts limit usually, as we want the full sparse cloud.
        """
        print(
            f"Loading sparse point cloud from COLMAP ({len(self.points3D_raw)} points)..."
        )

        xyz = []
        rgb = []

        for pid, p in self.points3D_raw.items():
            xyz.append(p.xyz)
            rgb.append(p.rgb / 255.0)

        xyz = np.array(xyz, dtype=np.float32)
        rgb = np.array(rgb, dtype=np.float32)
        normals = np.zeros_like(xyz)

        return BasicPointCloud(points=xyz, colors=rgb, normals=normals)
