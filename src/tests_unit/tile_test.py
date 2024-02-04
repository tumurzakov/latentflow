import torch
import unittest

from latentflow.tile import Tile,TileGenerator
from latentflow.latent import Latent

import logging
logging.basicConfig(level=logging.DEBUG)

class TestTile(unittest.TestCase):

    def test_should_generate_one_tile(self):
        latent = Latent(latent=torch.randn((1,4,2,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(1, len(gen.tiles))

        self.assertEqual(gen.tiles[0], (
                slice(0,2),
                slice(0,4),
                [0,1],
                slice(0,2),
                slice(0,2),
            ))

        self.assertEqual(1.0, gen.cover)

    def test_should_generate_two_tiles(self):
        latent = Latent(latent=torch.randn((1,4,4,2,2)))
        gen = TileGenerator(Tile(2,2,2), latent)
        self.assertEqual(2, len(gen.tiles))
        self.assertEqual(1.0, gen.cover)

    def test_should_cover_all_offset0(self):
        offset = 0
        latent = Latent(latent=torch.zeros((1,1,1,6,1)))
        gen = TileGenerator(Tile(
            length=1, length_overlap=0, length_offset=offset,
            height=3, height_overlap=0, height_offset=offset,
            width=1, width_overlap=0, width_offset=offset,
            ), latent)

        self.assertEqual(1.0, gen.cover)

    def test_should_cover_all_offset1(self):
        latent = Latent(latent=torch.zeros((1,1,1,6,1)))
        gen = TileGenerator(Tile(
            length=1, length_overlap=0, length_offset=0,
            height=3, height_overlap=0, height_offset=1,
            width=1, width_overlap=0, width_offset=0,
            ), latent)

        self.assertEqual(1.0, gen.cover)

    def test_should_cover_all_offset1_all(self):
        latent = Latent(latent=torch.zeros((1,1,6,6,6)))
        gen = TileGenerator(Tile(
            length=3, length_overlap=0, length_offset=1,
            height=3, height_overlap=0, height_offset=1,
            width=3, width_overlap=0, width_offset=1,
            ), latent)

        self.assertEqual(1.0, gen.cover)

    def _test_should_cover_all_offset1_all_overlap(self):
        latent = Latent(latent=torch.zeros((1,1,6,6,6)))
        gen = TileGenerator(Tile(
            length=3, length_overlap=1, length_offset=1,
            height=3, height_overlap=1, height_offset=1,
            width=3, width_overlap=1, width_offset=1,
            ), latent)

        self.assertEqual(1.0, gen.cover)


if __name__ == '__main__':
    unittest.main()
