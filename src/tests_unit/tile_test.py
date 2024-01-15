import torch
import unittest

from latentflow.tile import Tile,TileGenerator
from latentflow.latent import Latent

import logging
logging.basicConfig(level=logging.DEBUG)

class TestTile(unittest.TestCase):

    def _test_should_generate_one_tile(self):
        latent = Latent(latent=torch.randn((1,4,2,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(1, len(gen.tiles))

        latent = next(gen)
        self.assertEqual(latent, (
                slice(0,1),
                slice(0,4),
                slice(0,2),
                slice(0,2),
                slice(0,2),
            ))

    def _test_should_generate_two_tiles(self):
        latent = Latent(latent=torch.randn((1,4,4,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(2, len(gen.tiles))

        latent = next(gen)
        self.assertEqual(latent, (
                slice(0,1),
                slice(0,4),
                slice(0,2),
                slice(0,2),
                slice(0,2),
            ))

    def test_should_cover_all_offset0(self):
        offset = 0
        latent = Latent(latent=torch.zeros((1,1,1,6,1)))
        gen = TileGenerator(Tile(
            length=1, length_overlap=0,
            height=3, height_overlap=2,
            width=1, width_overlap=0,
            offset=offset
            ), latent)

        self.assertEqual(1.0, gen.cover)

    def test_should_cover_all_offset1(self):
        offset = 1
        latent = Latent(latent=torch.zeros((1,1,1,6,1)))
        gen = TileGenerator(Tile(
            length=1, length_overlap=0,
            height=3, height_overlap=2,
            width=1, width_overlap=0,
            offset=offset
            ), latent)

        self.assertEqual(1.0, gen.cover)



if __name__ == '__main__':
    unittest.main()
