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

### 4. Volumetric Fog Training Workflow

We recommend a 2-stage process for high-quality geometric separation between the object and the fog.

#### Step 1: Train Clean Surface
First, train the base object without any fog to get sharp geometry.
```bash
python src/scripts/train_volumetric.py --data_path data/nerf_synthetic/lego --output_path outputs/checkpoints/lego_clean_base --disable_fog
```

#### Step 2: Volumetric Fog Insertion
Now, freeze the surface geometry and force the model to learn a fog volume by simulating a foggy environment (blending training images with white).
```bash
python src/scripts/train_volumetric.py --data_path data/nerf_synthetic/lego --output_path outputs/checkpoints/lego_fog_final --pretrained_surface outputs/checkpoints/lego_clean_base/surface_final.pth --lock_surface --simulate_fog
```

### 5. Rendering & Comparisons

Generate a side-by-side comparison (Clean vs Fog 2x vs Fog 4x):
```bash
python src/scripts/render_volumetric.py --data_path data/nerf_synthetic/lego --surface_ckpt outputs/checkpoints/lego_clean_base/surface_final.pth --fog_ckpt outputs/checkpoints/lego_fog_final/fog_final.pth --output outputs/renders/comparison.gif --mode compare --compare_scales 2.0 4.0 --duration_ms 200
```

### 6. Export to PLY (Web Viewer Compatible)

Export the combined scene. Use `--prune_threshold` to remove invisible low-density fog points that clog up web viewers.
```bash
python src/scripts/export_ply.py --checkpoint outputs/checkpoints/lego_clean_base/surface_final.pth --fog_checkpoint outputs/checkpoints/lego_fog_final/fog_final.pth --output outputs/exports/lego_foggy.ply --fog_opacity_scale 3.0 --prune_threshold 0.05
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
