import torch
import numpy as np
import unittest
from src.models.gaussians import GaussianSet
from src.models.volumetric import VolumetricGaussianSet
from src.models.scene import Scene
from collections import namedtuple

# Mock BasicPointCloud
BasicPointCloud = namedtuple('BasicPointCloud', ['points', 'colors', 'normals'])

class TestPart2(unittest.TestCase):
    def setUp(self):
        pass

    def test_volumetric_gaussian_initialization(self):
        """Test VolumetricGaussianSet extends GaussianSet with density and phase color"""
        vgs = VolumetricGaussianSet(sh_degree=3)
        
        # Create dummy point cloud
        points = np.random.rand(50, 3)
        colors = np.random.rand(50, 3)
        normals = np.zeros((50, 3))
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        fog_density = 0.1
        fog_color = (0.8, 0.9, 1.0)
        vgs.create_from_pcd(pcd, fog_density=fog_density, fog_color=fog_color)
        
        # Test base Gaussian properties
        self.assertEqual(vgs.get_xyz.shape, (50, 3))
        self.assertEqual(vgs.get_opacity.shape, (50, 1))
        self.assertEqual(vgs.get_scaling.shape, (50, 3))
        
        # Test volumetric properties
        self.assertEqual(vgs.get_density.shape, (50, 1))
        self.assertEqual(vgs.get_phase_color.shape, (50, 3))
        
        # Verify density initialization
        self.assertAlmostEqual(vgs.get_density[0, 0].item(), fog_density, places=5)
        
        # Verify phase color initialization
        self.assertAlmostEqual(vgs.get_phase_color[0, 0].item(), fog_color[0], places=5)
        self.assertAlmostEqual(vgs.get_phase_color[0, 1].item(), fog_color[1], places=5)
        self.assertAlmostEqual(vgs.get_phase_color[0, 2].item(), fog_color[2], places=5)

    def test_volumetric_gaussian_setters(self):
        """Test density and phase color setters"""
        vgs = VolumetricGaussianSet(sh_degree=0)
        
        points = np.random.rand(10, 3)
        colors = np.random.rand(10, 3)
        normals = np.zeros((10, 3))
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        vgs.create_from_pcd(pcd)
        
        # Test set_density
        new_density = 0.5
        vgs.set_density(0, new_density)
        self.assertAlmostEqual(vgs.get_density[0, 0].item(), new_density, places=5)
        
        # Test set_phase_color
        new_color = (1.0, 0.0, 0.0)
        vgs.set_phase_color(0, new_color)
        self.assertAlmostEqual(vgs.get_phase_color[0, 0].item(), new_color[0], places=5)
        self.assertAlmostEqual(vgs.get_phase_color[0, 1].item(), new_color[1], places=5)
        self.assertAlmostEqual(vgs.get_phase_color[0, 2].item(), new_color[2], places=5)

    def test_scene_structure(self):
        """Test Scene class can hold both surface and volumetric Gaussians"""
        scene = Scene()
        
        # Create surface Gaussians
        surface_gs = GaussianSet(sh_degree=0)
        points_surface = np.random.rand(30, 3)
        colors_surface = np.random.rand(30, 3)
        normals_surface = np.zeros((30, 3))
        pcd_surface = BasicPointCloud(points=points_surface, colors=colors_surface, normals=normals_surface)
        surface_gs.create_from_pcd(pcd_surface)
        
        # Create volumetric Gaussians
        volume_gs = VolumetricGaussianSet(sh_degree=0)
        points_volume = np.random.rand(20, 3)
        colors_volume = np.random.rand(20, 3)
        normals_volume = np.zeros((20, 3))
        pcd_volume = BasicPointCloud(points=points_volume, colors=colors_volume, normals=normals_volume)
        volume_gs.create_from_pcd(pcd_volume)
        
        # Assign to scene
        scene.surface_gaussians = surface_gs
        scene.volume_gaussians = volume_gs
        
        # Verify scene contains both
        self.assertIsInstance(scene.surface_gaussians, GaussianSet)
        self.assertIsInstance(scene.volume_gaussians, VolumetricGaussianSet)
        self.assertEqual(scene.surface_gaussians.get_xyz.shape[0], 30)
        self.assertEqual(scene.volume_gaussians.get_xyz.shape[0], 20)

    def test_gradient_flow(self):
        """Test that gradients flow through volumetric parameters"""
        vgs = VolumetricGaussianSet(sh_degree=0)
        
        points = np.random.rand(5, 3)
        colors = np.random.rand(5, 3)
        normals = np.zeros((5, 3))
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        vgs.create_from_pcd(pcd)
        
        # Create a simple loss
        loss = (vgs.get_density ** 2).sum() + (vgs.get_phase_color ** 2).sum()
        loss.backward()
        
        # Check that gradients exist
        self.assertIsNotNone(vgs._density.grad)
        self.assertIsNotNone(vgs._phase_color.grad)
        self.assertTrue((vgs._density.grad != 0).any())
        self.assertTrue((vgs._phase_color.grad != 0).any())

if __name__ == '__main__':
    unittest.main()
