"""
Loss functions for Gaussian Splatting with Volumetric Effects.

Includes:
- L1 Loss: Simple photometric loss
- SSIM Loss: Structural similarity for perceptual quality
- Combined Photometric Loss: L1 + (1-SSIM)
- Fog Regularization: Sparsity constraint on volumetric density
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _ssim(img1, img2, window_size=11, size_average=True):
    """
    Compute SSIM between two images.
    
    Args:
        img1: (C, H, W) tensor
        img2: (C, H, W) tensor
        window_size: Size of gaussian window
        size_average: Return average or per-pixel SSIM
        
    Returns:
        SSIM value (scalar if size_average=True)
    """
    channel = img1.size(0)
    
    # Create gaussian window
    sigma = 1.5
    gauss = torch.Tensor([
        torch.exp(torch.tensor(-(x - window_size//2)**2 / float(2*sigma**2)))
        for x in range(window_size)
    ])
    gauss = gauss / gauss.sum()
    
    # Create 2D window
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)
    
    # Compute means
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=channel)
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Compute variances
    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=channel) - mu1_mu2
    
    # Constants for stability
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


class L1Loss(nn.Module):
    """Simple L1 (MAE) photometric loss."""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, rendered, target):
        """
        Args:
            rendered: (C, H, W) rendered image
            target: (C, H, W) target image
            
        Returns:
            L1 loss (scalar)
        """
        return F.l1_loss(rendered, target)


class SSIMLoss(nn.Module):
    """Structural Similarity loss (1 - SSIM)."""
    
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
    
    def forward(self, rendered, target):
        """
        Args:
            rendered: (C, H, W) rendered image
            target: (C, H, W) target image
            
        Returns:
            SSIM loss (scalar, 0 = perfect match)
        """
        ssim_value = _ssim(rendered, target, window_size=self.window_size)
        return 1.0 - ssim_value


class PhotometricLoss(nn.Module):
    """
    Combined photometric loss: L1 + λ_ssim * (1 - SSIM).
    
    This is commonly used in NeRF and Gaussian Splatting papers.
    """
    
    def __init__(self, lambda_ssim=0.2):
        super().__init__()
        self.lambda_ssim = lambda_ssim
        self.l1_loss = L1Loss()
        self.ssim_loss = SSIMLoss()
    
    def forward(self, rendered, target):
        """
        Args:
            rendered: (C, H, W) rendered image
            target: (C, H, W) target image
            
        Returns:
            Combined loss (scalar)
        """
        l1 = self.l1_loss(rendered, target)
        ssim = self.ssim_loss(rendered, target)
        return l1 + self.lambda_ssim * ssim


class FogRegularizationLoss(nn.Module):
    """
    Regularization for volumetric gaussians (fog).
    
    Encourages sparsity in fog density to prevent over-densification.
    Uses L1 penalty on density values.
    """
    
    def __init__(self, lambda_density=0.01):
        super().__init__()
        self.lambda_density = lambda_density
    
    def forward(self, volume_gaussians):
        """
        Args:
            volume_gaussians: VolumetricGaussianSet instance
            
        Returns:
            Sparsity loss on fog density (scalar)
        """
        if volume_gaussians is None or volume_gaussians._xyz.shape[0] == 0:
            return torch.tensor(0.0, device=volume_gaussians._xyz.device if volume_gaussians else torch.device('cpu'))
        
        # L1 penalty on density to encourage sparsity
        density = volume_gaussians._density
        sparsity_loss = torch.abs(density).mean()
        
        return self.lambda_density * sparsity_loss


class TotalLoss(nn.Module):
    """
    Complete loss combining photometric loss and fog regularization.
    
    Total = Photometric(rendered, target) + FogRegularization(volumes)
    """
    
    def __init__(self, lambda_ssim=0.2, lambda_density=0.01):
        super().__init__()
        self.photometric_loss = PhotometricLoss(lambda_ssim=lambda_ssim)
        self.fog_regularization = FogRegularizationLoss(lambda_density=lambda_density)
    
    def forward(self, rendered, target, volume_gaussians=None):
        """
        Args:
            rendered: (C, H, W) rendered image
            target: (C, H, W) target image
            volume_gaussians: Optional VolumetricGaussianSet for regularization
            
        Returns:
            Total loss (scalar)
        """
        photo_loss = self.photometric_loss(rendered, target)
        
        if volume_gaussians is not None:
            fog_reg = self.fog_regularization(volume_gaussians)
            return photo_loss + fog_reg
        
        return photo_loss