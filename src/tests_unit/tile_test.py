import torch
import unittest

from latentflow.tile import Tile,TileGenerator
from latentflow.latent import Latent

class TestTile(unittest.TestCase):

    def test_should_generate_one_tile(self):
        latent = Latent(latent=torch.randn((1,4,2,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(1, len(gen.tiles))

        latent = next(gen)
        self.assertTrue(isinstance(latent, Latent))
        self.assertEqual(latent.latent.shape, torch.Size((1,4,2,2,2)))

    def test_should_generate_two_tiles(self):
        latent = Latent(latent=torch.randn((1,4,4,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(2, len(gen.tiles))

        latent = next(gen)
        self.assertTrue(isinstance(latent, Latent))
        self.assertEqual(latent.latent.shape, torch.Size((1,4,2,2,2)))


