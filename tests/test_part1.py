import torch
import numpy as np
import unittest
from src.models.gaussians import GaussianSet
from src.datasets.blender import BlenderDataset
from collections import namedtuple
import os

# Mock BasicPointCloud
BasicPointCloud = namedtuple('BasicPointCloud', ['points', 'colors', 'normals'])

class TestPart1(unittest.TestCase):
    def setUp(self):
        pass

    def test_gaussian_initialization(self):
        gs = GaussianSet(sh_degree=3)
        
        # Create dummy point cloud
        points = np.random.rand(100, 3)
        colors = np.random.rand(100, 3)
        normals = np.zeros((100, 3))
        pcd = BasicPointCloud(points=points, colors=colors, normals=normals)
        
        gs.create_from_pcd(pcd)
        
        self.assertEqual(gs.get_xyz.shape, (100, 3))
        self.assertEqual(gs.get_opacity.shape, (100, 1))
        self.assertEqual(gs.get_scaling.shape, (100, 3))
        # Features: (N, K, 3) where K is the number of SH coefficients
        self.assertEqual(gs.get_features.shape, (100, 16, 3))

    def test_blender_dataset(self):
        # Assumes dummy_data exists from shell commands
        if not os.path.exists("dummy_data"):
             self.skipTest("dummy_data not found")
        
        dataset = BlenderDataset(data_path="dummy_data", split="train")
        self.assertEqual(len(dataset), 1)
        
        cam = dataset[0]
        self.assertEqual(cam.image.shape, (100, 100, 3))
        self.assertAlmostEqual(cam.FovX, 0.8)
        
        # Test point cloud generation
        pcd = dataset.get_point_cloud()
        self.assertEqual(pcd.points.shape[0], 100000)

if __name__ == '__main__':
    unittest.main()
