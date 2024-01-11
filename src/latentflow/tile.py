import torch
import logging

from .latent import Latent

class Tile:
    def __init__(self,
            height: int,
            width: int,
            length: int,
            width_overlap:int=0,
            height_overlap:int=0,
            length_overlap:int=0):

        self.width = width
        self.height = height
        self.length = length
        self.width_overlap = width_overlap
        self.height_overlap = height_overlap
        self.length_overlap = length_overlap

        logging.debug(f'{self}')

    def __str__(self):
        return "Tile(%s-%s %s-%s %s-%s)" % (
                self.width, self.width_overlap,
                self.height, self.height_overlap,
                self.length, self.length_overlap,
                )

class TileGenerator:
    def __init__(self, tile: Tile, latent: Latent):
        logging.debug("TileGenerator init %s %s", tile, latent)

        self.tile = tile
        self.latent = latent

        self._index = 0
        self.tiles = self.simple()

        logging.debug("TileGenerator tiles %s", len(self.tiles))


    def simple(self):
        shape = self.latent.latent.shape

        tiles = []
        for b in range(0, shape[0]):
            for l in range(0, shape[2], self.tile.length):
                for h in range(0, shape[3], self.tile.height):
                    for w in range(0, shape[4], self.tile.width):
                        tiles.append((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l, l + self.tile.length),
                            slice(h, h + self.tile.height),
                            slice(w, w + self.tile.width),
                            ))

        return tiles


    def __len__(self):
        return len(self.tiles) if self.tiles is not None else 0

    def __str__(self):
        return f'tiles({len(self)})'

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.tiles):
            tile = self.tiles[self._index]
            logging.debug("TileGenerator next %s %s", self._index, tile)
            self._index += 1

            return tile
        else:
            raise StopIteration
