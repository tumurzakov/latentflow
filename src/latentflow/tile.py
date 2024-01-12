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
            length_overlap:int=0,
            offset:int=0
            ):


        self.width = width
        self.height = height
        self.length = length
        self.width_overlap = width_overlap
        self.height_overlap = height_overlap
        self.length_overlap = length_overlap
        self.offset = offset

        logging.debug(f'{self}')

    def __str__(self):
        return "Tile(%s-%s %s-%s %s-%s, %s)" % (
                self.width, self.width_overlap,
                self.height, self.height_overlap,
                self.length, self.length_overlap,
                self.offset
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

            l_start = self.tile.offset
            l_end = self.tile.length + self.tile.offset

            while l_end <= shape[2]:

                h_start = self.tile.offset
                h_end = self.tile.height + self.tile.offset

                while h_end <= shape[3]:

                    w_start = self.tile.offset
                    w_end = self.tile.width + self.tile.offset

                    while w_end <= shape[4]:

                        tiles.append((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l_start, l_end),
                            slice(h_start, h_end),
                            slice(w_start, w_end),
                            ))

                        w_start += self.tile.width_overlap
                        w_end += self.tile.width_overlap

                    h_start += self.tile.height_overlap
                    h_end += self.tile.height_overlap

                l_start += self.tile.length_overlap
                l_end += self.tile.length_overlap

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
