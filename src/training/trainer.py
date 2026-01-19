"""
Trainer for Gaussian Splatting with Volumetric Effects.

Provides flexible training loop with:
- Support for surface and volumetric gaussians
- Ability to freeze/unfreeze surface or volume parameters
- Configurable loss functions
- Logging and checkpointing
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
import os
from tqdm import tqdm

from src.training.losses import TotalLoss
from src.rendering.rasterizer import render_gaussians
from src.rendering.composition import composite_surface_and_volume


class Trainer:
    """
    Generic trainer for Gaussian Splatting with optional volumetric effects.
    
    Supports:
    - Training surface gaussians only
    - Training volumetric gaussians only (with frozen surfaces)
    - Joint training of both
    """
    
    def __init__(
        self,
        scene,
        optimizer,
        loss_fn=None,
        device='cpu',
        output_dir='outputs',
        log_interval=10,
        save_interval=100
    ):
        """
        Args:
            scene: Scene object containing surface and/or volume gaussians
            optimizer: torch.optim optimizer
            loss_fn: Loss function (defaults to TotalLoss)
            device: 'cpu', 'cuda', or 'mps'
            output_dir: Directory for checkpoints and logs
            log_interval: Log every N iterations
            save_interval: Save checkpoint every N iterations
        """
        self.scene = scene
        self.optimizer = optimizer
        self.loss_fn = loss_fn if loss_fn is not None else TotalLoss()
        self.device = device
        self.output_dir = output_dir
        self.log_interval = log_interval
        self.save_interval = save_interval
        
        # Move scene to device
        self.scene = self.scene.to(device)
        
        # Setup directories
        os.makedirs(f"{output_dir}/checkpoints", exist_ok=True)
        os.makedirs(f"{output_dir}/logs", exist_ok=True)
        
        # Training state
        self.iteration = 0
        self.losses = []
    
    def freeze_surface_gaussians(self):
        """Freeze surface gaussian parameters (only optimize volumes)."""
        if hasattr(self.scene, 'surface_gaussians'):
            for gaussians in self.scene.surface_gaussians:
                for param in gaussians.parameters():
                    param.requires_grad = False
        print("✓ Surface gaussians frozen")
    
    def unfreeze_surface_gaussians(self):
        """Unfreeze surface gaussian parameters."""
        if hasattr(self.scene, 'surface_gaussians'):
            for gaussians in self.scene.surface_gaussians:
                for param in gaussians.parameters():
                    param.requires_grad = True
        print("✓ Surface gaussians unfrozen")
    
    def freeze_volume_gaussians(self):
        """Freeze volumetric gaussian parameters (only optimize surfaces)."""
        if hasattr(self.scene, 'volume_gaussians'):
            for gaussians in self.scene.volume_gaussians:
                for param in gaussians.parameters():
                    param.requires_grad = False
        print("✓ Volume gaussians frozen")
    
    def unfreeze_volume_gaussians(self):
        """Unfreeze volumetric gaussian parameters."""
        if hasattr(self.scene, 'volume_gaussians'):
            for gaussians in self.scene.volume_gaussians:
                for param in gaussians.parameters():
                    param.requires_grad = True
        print("✓ Volume gaussians unfrozen")
    
    def render_scene(self, camera, bg_color=None):
        """
        Render the scene from a camera viewpoint.
        
        Args:
            camera: Camera object
            bg_color: Background color (3,) tensor
            
        Returns:
            Rendered image (C, H, W) tensor
        """
        if bg_color is None:
            bg_color = torch.zeros(3, device=self.device)
        
        # Check if scene has volumetric effects
        has_volumes = (
            hasattr(self.scene, 'volume_gaussians') and 
            len(self.scene.volume_gaussians) > 0
        )
        
        if has_volumes:
            # Render with volumetric composition
            return composite_surface_and_volume(
                self.scene,
                camera,
                bg_color=bg_color
            )
        else:
            # Render surface only
            if not hasattr(self.scene, 'surface_gaussians') or len(self.scene.surface_gaussians) == 0:
                # No gaussians to render
                H, W = camera.height, camera.width
                return bg_color.view(3, 1, 1).expand(3, H, W)
            
            # Get first surface gaussian set
            surface_gaussians = self.scene.surface_gaussians[0]
            return render_gaussians(
                surface_gaussians,
                camera,
                bg_color=bg_color
            )
    
    def train_step(self, camera, target_image):
        """
        Single training step: forward -> loss -> backward -> optimizer step.
        
        Args:
            camera: Camera object for rendering
            target_image: Target RGB image (C, H, W) tensor
            
        Returns:
            Loss value (scalar)
        """
        # Zero gradients
        self.optimizer.zero_grad()
        
        # Forward pass: render scene
        rendered = self.render_scene(camera)
        
        # Compute loss
        # Get volume gaussians for regularization if present
        volume_gaussians = None
        if hasattr(self.scene, 'volume_gaussians') and len(self.scene.volume_gaussians) > 0:
            volume_gaussians = self.scene.volume_gaussians[0]
        
        loss = self.loss_fn(rendered, target_image, volume_gaussians=volume_gaussians)
        
        # Backward pass
        loss.backward()
        
        # Optimizer step
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, dataloader, num_iterations):
        """
        Main training loop.
        
        Args:
            dataloader: Iterator providing (camera, target_image) pairs
            num_iterations: Number of training iterations
        """
        print(f"Starting training for {num_iterations} iterations...")
        print(f"Device: {self.device}")
        print(f"Output directory: {self.output_dir}")
        
        # Iterate through data
        print(dataloader[0])
        data_iter = iter(dataloader)
        
        bar = tqdm(range(1, num_iterations + 1))
        for it in bar:
            self.iteration = it
            
            # Get next batch (cycle through dataset)
            try:
                camera, target_image = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                camera, target_image = next(data_iter)
            
            # Move to device
            if hasattr(camera, 'to'):
                camera = camera.to(self.device)
            if isinstance(target_image, torch.Tensor):
                target_image = target_image.to(self.device)
            
            # Training step
            loss = self.train_step(camera, target_image)
            self.losses.append(loss)
            
            # Logging
            if it % self.log_interval == 0:
                avg_loss = sum(self.losses[-self.log_interval:]) / len(self.losses[-self.log_interval:])
                print(f"[Iter {it:05d}/{num_iterations}] Loss = {loss:.6f} (avg: {avg_loss:.6f})")
            
            # Checkpointing
            if it % self.save_interval == 0:
                self.save_checkpoint(f"iter_{it:05d}.pth")
            bar.set_description(f"Iter: {it}, loss: {loss.item():.4f}")
        
        # Save final checkpoint
        self.save_checkpoint("final.pth")
        print("✓ Training complete!")
    
    def save_checkpoint(self, filename):
        """Save model checkpoint."""
        checkpoint_path = os.path.join(self.output_dir, "checkpoints", filename)
        
        checkpoint = {
            'iteration': self.iteration,
            'scene_state_dict': self.scene.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'losses': self.losses,
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"✓ Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.scene.load_state_dict(checkpoint['scene_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint['iteration']
        self.losses = checkpoint['losses']
        
        print(f"✓ Checkpoint loaded: {checkpoint_path} (iteration {self.iteration})")


def create_trainer(
    scene,
    learning_rate=1e-3,
    lambda_ssim=0.2,
    lambda_density=0.01,
    device='cpu',
    **kwargs
):
    """
    Convenience function to create a trainer with standard configuration.
    
    Args:
        scene: Scene object
        learning_rate: Learning rate for Adam optimizer
        lambda_ssim: Weight for SSIM loss
        lambda_density: Weight for fog density regularization
        device: Device to train on
        **kwargs: Additional arguments passed to Trainer
        
    Returns:
        Configured Trainer instance
    """
    # Create optimizer
    optimizer = torch.optim.Adam(scene.parameters(), lr=learning_rate)
    
    # Create loss function
    loss_fn = TotalLoss(lambda_ssim=lambda_ssim, lambda_density=lambda_density)
    
    # Create trainer
    trainer = Trainer(
        scene=scene,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=device,
        **kwargs
    )
    
    return trainer
