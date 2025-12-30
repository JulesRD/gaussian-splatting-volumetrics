# Projects Gaussians to screen
#
# Computes 2D Gaussian splats
#
# Shared between surface & volume
#
# Do not mix compositing logic here.
import torch

from src.rendering.camera import project_points


def gaussian_2d(
    dx: torch.Tensor,
    dy: torch.Tensor,
    sigma_x: torch.Tensor,
    sigma_y: torch.Tensor,
):
    """
    Computes unnormalized 2D Gaussian.
    """
    return torch.exp(
        -0.5 * ((dx / sigma_x) ** 2 + (dy / sigma_y) ** 2)
    )

def render_gaussians(
    gaussians,
    camera,
    bg_color: torch.Tensor,
):
    """
    Naive Gaussian splatting renderer.

    Args:
        gaussians: GaussianSet
        camera: CameraInfo
        bg_color: (3,) tensor

    Returns:
        image: (3, H, W)
    """
    device = gaussians.get_xyz.device
    H, W = camera.height, camera.width

    # Output buffers
    image = torch.zeros((3, H, W), device=device)
    alpha = torch.zeros((H, W), device=device)

    # Project Gaussians
    xy, depth = project_points(gaussians.get_xyz, camera)

    colors = gaussians.get_features[:, 0, :]
    opacity = gaussians.get_opacity.squeeze(-1)
    scale = gaussians.get_scaling.mean(dim=1)

    # Sort front-to-back
    sorted_idx = torch.argsort(depth)
    xy = xy[sorted_idx]
    depth = depth[sorted_idx]
    colors = colors[sorted_idx]
    opacity = opacity[sorted_idx]
    scale = scale[sorted_idx]

    # Pixel grid
    ys = torch.arange(H, device=device)
    xs = torch.arange(W, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")

    for i in range(xy.shape[0]):
        x0, y0 = xy[i]

        # Bounding box (3 sigma)
        sigma = scale[i] * 50.0  # heuristic scale → pixels
        if sigma < 1e-3:
            continue

        xmin = int(max(0, x0 - 3 * sigma))
        xmax = int(min(W - 1, x0 + 3 * sigma))
        ymin = int(max(0, y0 - 3 * sigma))
        ymax = int(min(H - 1, y0 + 3 * sigma))

        if xmin >= xmax or ymin >= ymax:
            continue

        dx = grid_x[ymin:ymax, xmin:xmax] - x0
        dy = grid_y[ymin:ymax, xmin:xmax] - y0

        weight = gaussian_2d(dx, dy, sigma, sigma)
        a = opacity[i] * weight

        # Alpha compositing
        one_minus_alpha = 1.0 - alpha[ymin:ymax, xmin:xmax]
        contrib = a * one_minus_alpha

        image[:, ymin:ymax, xmin:xmax] += (
            colors[i].view(3, 1, 1) * contrib
        )

        alpha[ymin:ymax, xmin:xmax] += contrib

    # Background
    image += bg_color.view(3, 1, 1) * (1.0 - alpha)

    return image.clamp(0.0, 1.0)


if __name__ == "__main__":
    """
    Minimal rasterizer sanity check.
    Renders a single Gaussian blob.
    """
    import math
    class DummyCamera:
        width = 256
        height = 256
        FovX = FovY = math.radians(60)
        R = torch.eye(3).numpy()
        T = torch.zeros(3).numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fake GaussianSet-like object
    class DummyGaussians:
        def __init__(self):
            self.xyz = torch.tensor([[0.0, 0.0, 2.0]], device=device)
            self.scale = torch.tensor([[0.05, 0.05, 0.05]], device=device)
            self.color = torch.tensor([[1.0, 0.0, 0.0]], device=device)
            self.opacity = torch.tensor([[0.8]], device=device)

        @property
        def get_xyz(self): return self.xyz
        @property
        def get_scaling(self): return self.scale
        @property
        def get_features(self): return self.color.view(1, 3, 1)
        @property
        def get_opacity(self): return self.opacity

    gaussians = DummyGaussians()
    camera = DummyCamera()
    bg = torch.zeros(3, device=device)

    img = render_gaussians(gaussians, camera, bg)

    print("Rasterizer test image:", img.shape, img.min().item(), img.max().item())
