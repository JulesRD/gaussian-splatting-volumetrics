from src.models.gaussians import GaussianSet

class VolumetricGaussianSet(GaussianSet):
    def __init__(self):
        super().__init__()
        self.density        # replaces opacity semantics
        self.phase_color    # scattering color