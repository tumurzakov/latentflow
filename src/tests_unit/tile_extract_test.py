import torch

from latentflow.tile import Tile
from latentflow.latent import Latent
from latentflow.tile_extract import TileExtract

class TestTileExtract(TileExtract):
    def apply(self, latent: Latent) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))
