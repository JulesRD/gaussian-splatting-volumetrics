# Gaussian Splatting Volumetrics

Extend Gaussian Splatting to model semi-transparent media (e.g. fog or smoke).

## Getting Started

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv .venv
# Activate
source .venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

or with uv

```bash
uv sync
source .venv/bin/activate
```

### 2. Download Real Data (NeRF + COLMAP)
```bash
./get_data.sh
```

### 3. Run Part 1 Demo (Data & Initialization)
This script loads the dataset and initializes the 3D Gaussian field as a sparse point cloud.

```bash
export PYTHONPATH=$PYTHONPATH:.
# Run with the real Lego dataset
python3 src/scripts/part1_demo.py data/nerf_synthetic/lego
```

### 4. Train a scene
```bash
python src/scripts/train.py --num_iter 7000 --stop_dense_after 7000 --dense_th 0.0002 --min_opacity 0.01 --reset_interval 3000 --data_path data/drjohnson
```

with volumetrics:
```bash
python src/scripts/train_volumetric.py --num_iter 7000 --stop_dense_after 7000 --init_points 50000 --dense_th 0.0002 --min_opacity 0.01 --reset_interval 3000 --data_path data/nerf_synthetic/lego
```

then generate the `.ply` file:
```bash
python src/scripts/export_ply.py --checkpoint outputs/checkpoints/drjohnson/gaussians_final_7000_761557.pth --output outputs/point_cloud/drjohnson.ply
```

with volumetrics:
```bash
python src/scripts/export_ply.py --checkpoint outputs/checkpoints/lego_volumetric/surface_final.pth --fog_checkpoint outputs/checkpoints/lego_volumetric/fog_final.pth --output outputs/point_cloud/lego.ply
```


### 4. Visualization
After running the training, a `.ply` file is generated in the `point_cloud` directory.
- Open this file in any 3DGS viewer (ex. https://storysplat.com/editor), **Blender** with Gaussian Splatting addon

It is possible to generate the generation timeline:
```bash
python src/scripts/progression.py data/truck --checkpoints outputs/checkpoints/truck --out outputs/renders/truck.gif --camera 6
```

## Project Structure
- `src/models`: Core Gaussian and Volumetric definitions.
- `src/datasets`: Data loaders (Blender/NeRF format).
- `src/rendering`: Projection and rasterization logic.
- `src/training`: Trainer and loss functions.
- `src/scripts`: Entry points for training, rendering, and demos.
