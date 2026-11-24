import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import torch
from torch import nn

from mambagir.vision import VisualEncoder


class SpatialBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 8, 1)
        self.levels = nn.ModuleList([nn.Identity()])
        self.norm = nn.BatchNorm2d(8)


class VisionTests(unittest.TestCase):
    def test_preserves_49_spatial_tokens_and_global_feature_uses_same_encoder(self):
        encoder = VisualEncoder(backbone=SpatialBackbone()).eval()
        images = torch.randn(2, 3, 14, 14)
        patches = encoder.forward_patches(images)
        self.assertEqual(tuple(patches.shape), (2, 49, 8))
        self.assertTrue(torch.allclose(patches.mean(1), encoder.forward_global(images), atol=1e-6))
        self.assertFalse(torch.allclose(patches[:, 0], patches[:, -1]))

    def test_pretrained_loading_rejects_wrong_encoder_instead_of_silently_ignoring(self):
        encoder = VisualEncoder(backbone=SpatialBackbone())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'wrong.pt'
            torch.save({'state_dict': {'wrong.weight': torch.ones(1)}}, checkpoint)
            with self.assertRaises(RuntimeError):
                encoder.load_pretrained(checkpoint)

    def test_official_checkpoint_namespace_metadata_does_not_block_weights(self):
        original = SpatialBackbone()
        encoder = VisualEncoder(backbone=SpatialBackbone())
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'official.pt'
            torch.save({'state_dict': original.state_dict(), 'args': Namespace(model='MambaVision')}, checkpoint)
            encoder.load_pretrained(checkpoint)
        self.assertTrue(torch.equal(encoder.backbone.patch_embed.weight, original.patch_embed.weight))


if __name__ == '__main__':
    unittest.main()
