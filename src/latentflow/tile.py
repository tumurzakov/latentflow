import torch
import logging
import random

from .latent import Latent
from .tensor import Tensor

class Tile:
    def __init__(self,
            length: int,
            height: int,
            width: int,
            width_overlap:int=0,
            height_overlap:int=0,
            length_overlap:int=0,
            length_offset:int=0,
            offset:int=0
            ):

        self.length = length
        self.length_overlap = length_overlap
        if self.length_overlap == 0:
            self.length_overlap = length

        self.width = width
        self.width_overlap = width_overlap
        if self.width_overlap == 0:
            self.width_overlap = width

        self.height = height
        self.height_overlap = height_overlap
        if self.height_overlap == 0:
            self.height_overlap = height

        self.offset = offset

        logging.debug(f'{self}')

    def __str__(self):
        return "Tile(l%s->%s h%s->%s w%s->%s, %s)" % (
                self.length, self.length_overlap,
                self.height, self.height_overlap,
                self.width, self.width_overlap,
                self.offset
                )

class TileGenerator:
    def __init__(self,
            tile: Tile,
            latent: Latent,
            strategy='simple',
            do_classifier_free_guidance=True,
            *args,
            **kwargs):

        logging.debug("TileGenerator init %s %s", tile, latent)

        self.tile = tile
        self.latent = latent

        self.tiles = []

        self.audit = torch.zeros_like(latent.latent).int()

        self.do_classifier_free_guidance = do_classifier_free_guidance

        if strategy == 'random':
            self.simple(Tile(
                length=self.tile.length,
                height=self.tile.height,
                width=self.tile.width,
                # no overlap and offset
                ))
            self.audit = torch.zeros_like(latent.latent).int()
            self.random(self.tile, *args, **kwargs)
        elif strategy == 'simple' and self.tile.offset > 0:
            self.simple(Tile(
                length=self.tile.length,
                height=self.tile.height,
                width=self.tile.width,
                # no overlap and offset
                ))
            self.audit = torch.zeros_like(latent.latent).int()
            self.simple(self.tile)
        else:
            self.simple(self.tile)

        self._index = 0

        self.cover = self.audit.sum() / self.audit.numel()
        logging.debug("TileGenerator tiles [%s] cover=%s",
                len(self.tiles), self.cover.item())

    def append_tile(self, tile):
        if self.audit[tile].numel() > 0 \
            and self.audit[tile].sum() != self.audit[tile].numel():

            if self.do_classifier_free_guidance:
                tile = (
                        slice(tile[0].start*2, tile[0].stop*2),
                        tile[1],
                        tile[2],
                        tile[3],
                        tile[4],
                        )

            self.tiles.append(tile)
            self.audit[tile] = 1

    def random(self, tile, random_threshold=0.5, random_max_steps=1000, random_min_pixel_count=1000):
        logging.debug("TileGenerator random %s %s", tile, random_threshold)

        shape = self.latent.latent.shape
        length = shape[2]

        cnt = 0

        audit_numel = self.audit.numel()

        while True:
            b = random.randrange(shape[0])
            l = random.randrange(shape[2])
            h = random.randrange(shape[3]-tile.height)
            w = random.randrange(shape[4]-tile.width)
            t = (
                slice(b, b + 1),
                slice(0, 4),
                [x%length for x in range(l, l+self.tile.length)],
                slice(h, h+self.tile.height),
                slice(w, w+self.tile.width),
                )

            tile_sum = self.audit[t].sum()
            tile_numel = self.audit[t].numel()

            if tile_numel > 0 and tile_numel - tile_sum > random_min_pixel_count:
                self.append_tile(t)

            audit_sum = self.audit.sum().item()

            cnt += 1

            if audit_sum / audit_numel > random_threshold or cnt > random_max_steps:
                break

        logging.info("GenerateTiles random %s %s", len(self.tiles), audit_sum/audit_numel)

    def simple(self, tile):
        logging.debug("TileGenerator simple %s", tile)

        shape = self.latent.latent.shape

        offset = tile.offset

        for b in range(0, shape[0]):

            l_start = offset
            l_end = tile.length + offset

            while l_start <= shape[2]:

                h_start = offset
                h_end = tile.height + offset

                while h_start <= shape[3]:

                    w_start = offset
                    w_end = tile.width + offset

                    while w_start <= shape[4]:

                        self.append_tile((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l_start, l_end),
                            slice(h_start, h_end),
                            slice(w_start, w_end),
                            ))

                        w_start += tile.width_overlap
                        w_end += tile.width_overlap

                    if w_start < shape[4] and w_end > shape[4]:
                        self.append_tile((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l_start, l_end),
                            slice(h_start, h_end),
                            slice(w_start, w_end),
                            ))

                    h_start += tile.height_overlap
                    h_end += tile.height_overlap

                if h_start < shape[3] and h_end > shape[3]:
                    self.append_tile((
                        slice(b, b + 1),
                        slice(0, 4),
                        slice(l_start, l_end),
                        slice(h_start, h_end),
                        slice(w_start, w_end),
                        ))

                l_start += tile.length_overlap
                l_end += tile.length_overlap

            if l_start < shape[2] and l_end > shape[2]:
                self.append_tile((
                    (b, b + 1),
                    (0, 4),
                    (l_start, l_end),
                    (h_start, h_end),
                    (w_start, w_end),
                    ))

        logging.info("GenerateTiles simple %s", len(self.tiles))

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
