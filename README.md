# Gaussian Splatting Volumetrics

Extend Gaussian Splatting to model semi-transparent media (e.g. fog or smoke).

## Getting Started

### 1. Setup Environment
```bash
# Create virtual environment
python3 -m venv venv
# Activate
source venv/bin/activate
# Install dependencies
pip install -r requirements.txt
```

### 2. Download Real Data (Lego)
```bash
mkdir -p data
curl -L -o data/nerf_example_data.zip http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/ECCV20/nerf/nerf_example_data.zip
unzip -q data/nerf_example_data.zip -d data/
rm data/nerf_example_data.zip
```

### 3. Run Part 1 Demo (Data & Initialization)
This script loads the dataset and initializes the 3D Gaussian field as a sparse point cloud.

```bash
export PYTHONPATH=$PYTHONPATH:.
# Run with the real Lego dataset
python3 src/scripts/part1_demo.py data/nerf_synthetic/lego
```

### 4. Visualization
After running the demo, an `initial_state.ply` file is generated in the root directory.
- Open this file in **MeshLab**, **Blender**, or any 3D viewer.
- You will see a 3D cloud of points (100,000 points) initialized within the camera frustums of the Lego scene.
- Each point represents an initial Gaussian seed with position, color (SH), and opacity.

## Project Structure
- `src/models`: Core Gaussian and Volumetric definitions.
- `src/datasets`: Data loaders (Blender/NeRF format).
- `src/rendering`: Projection and rasterization logic.
- `src/training`: Trainer and loss functions.
- `src/scripts`: Entry points for training, rendering, and demos.