import torch

from .tile import Tile
from .latent import Latent

class TileExtract:
    def __init__(self, tile: Tile):
        self.tile = tile

    def apply(self, latent: Latent) -> Latent:
        pass
