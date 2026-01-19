import bpy
import json
import math
import os
from mathutils import Matrix

print("Hello")
# ==============================
# USER SETTINGS
# ==============================
OUTPUT_PATH = bpy.path.abspath("//transforms.json")
IMAGE_FOLDER = "images"   # relative path used in transforms.json
CAMERA_NAME = bpy.context.scene.camera.name

# ==============================
# HELPER FUNCTIONS
# ==============================

def blender_to_nerf_matrix(cam_matrix):
    """
    Standard OpenGL convention (matches NeRF).
    Blender camera local axes: -Z forward, +Y up.
    """
    return cam_matrix


def get_camera_intrinsics(scene, cam):
    """
    Compute camera intrinsics in NeRF format
    """
    render = scene.render
    width = render.resolution_x
    height = render.resolution_y
    scale = render.resolution_percentage / 100.0

    cam_data = cam.data
    focal_length_mm = cam_data.lens
    sensor_width_mm = cam_data.sensor_width

    focal_px = (focal_length_mm / sensor_width_mm) * width

    return {
        "camera_angle_x": 2 * math.atan(width / (2 * focal_px)),
        "fl_x": focal_px,
        "fl_y": focal_px,
        "cx": width / 2,
        "cy": height / 2,
        "w": int(width * scale),
        "h": int(height * scale)
    }

# ==============================
# MAIN EXPORT
# ==============================

scene = bpy.context.scene
cam = bpy.data.objects[CAMERA_NAME]

frames = []

start = scene.frame_start
end = scene.frame_end

for frame in range(start, end + 1):
    scene.frame_set(frame)

    cam_matrix = cam.matrix_world
    nerf_matrix = blender_to_nerf_matrix(cam_matrix)

    frames.append({
        "file_path": f"{IMAGE_FOLDER}/{frame:04d}.png",
        "transform_matrix": [
            list(row) for row in nerf_matrix
        ]
    })

intrinsics = get_camera_intrinsics(scene, cam)

transforms = {
    **intrinsics,
    "frames": frames
}

with open(OUTPUT_PATH, "w") as f:
    json.dump(transforms, f, indent=4)

print(f"[INFO] Camera poses exported to {OUTPUT_PATH}")
