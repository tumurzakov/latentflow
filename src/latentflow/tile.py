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
            height_offset:int=0,
            width_offset:int=0,
            ):

        self.length = length
        self.length_overlap = length_overlap
        self.length_offset = length_offset

        self.width = width
        self.width_overlap = width_overlap
        self.width_offset = width_offset

        self.height = height
        self.height_overlap = height_overlap
        self.height_offset = height_offset


        logging.debug(f'{self}')

    def __str__(self):
        return "Tile(l%s->%s+%s h%s->%s+%s w%s->%s+%s)" % (
                self.length, self.length_overlap, self.length_offset,
                self.height, self.height_overlap, self.height_offset,
                self.width, self.width_overlap, self.width_offset,
                )

class TileGenerator:
    def __init__(self,
            tile: Tile,
            latent: Latent,
            do_classifier_free_guidance=True,
            pixel_infer_count=None,
            *args,
            **kwargs):

        logging.debug("TileGenerator init %s %s", tile, latent)

        self.tile = tile
        self.latent = latent
        self.latent_shape = self.latent.latent.shape

        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.pixel_infer_count = pixel_infer_count
        if self.pixel_infer_count is None:
            self.pixel_infer_count = torch.zeros_like(latent.latent)
        else:
            assert isinstance(self.pixel_infer_count, Tensor), "pixel_infer_count should be Tensor"
            self.pixel_infer_count = self.pixel_infer_count.tensor

        self.tiles = []
        self.audit = torch.zeros_like(latent.latent).int()

        self.loop_control = 1000
        self.simple(self.tile)

        self._index = 0

        self.cover = self.audit.sum() / self.audit.numel()
        logging.debug("TileGenerator tiles [%s] cover=%s",
                len(self.tiles), self.cover.item())

    def append_tile(self, tile):
        shape = self.latent_shape

        lslice = tile[2]
        if lslice.stop > shape[2]:
            diff = lslice.stop - shape[2]
            lslice = slice(lslice.start-diff, lslice.stop-diff)

            tile = (
                    tile[0],
                    tile[1],
                    lslice,
                    tile[3],
                    tile[4],
                    )

        if self.audit[tile].numel() > 0 \
            and self.audit[tile].sum() != self.audit[tile].numel():

            bslice = tile[0]
            if self.do_classifier_free_guidance:
                bslice = slice(tile[0].start*2, tile[0].stop*2)
                tile = (
                        bslice,
                        tile[1],
                        lslice,
                        tile[3],
                        tile[4],
                        )

            self.tiles.append(tile)
            self.audit[tile] = 1
            self.pixel_infer_count[tile] += 1

    def simple(self, tile):
        logging.debug("TileGenerator simple %s", tile)

        shape = self.latent.latent.shape

        for b in range(0, shape[0]):

            l_start = 0
            l_end = tile.length

            while l_start <= shape[2]:

                h_start = 0
                h_end = tile.height

                while h_start <= shape[3]:

                    w_start = 0
                    w_end = tile.width

                    while w_start <= shape[4]:
                        self.loop_control -= 1
                        if self.loop_control < 0:
                            logging.error("Infinite loop detected l%s-%s, h%s-%s, w%s-%s",
                                    l_start, l_end, h_start, h_end, w_start, w_end)
                            raise Exception("Infinite loop detected")

                        self.append_tile((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l_start, l_end),
                            slice(h_start, h_end),
                            slice(w_start, w_end),
                            ))

                        overlap = tile.width - tile.width_overlap
                        offset = overlap - tile.length_offset % overlap
                        w_start += offset
                        w_end += offset

                    if w_start < shape[4] and w_end > shape[4]:
                        self.append_tile((
                            slice(b, b + 1),
                            slice(0, 4),
                            slice(l_start, l_end),
                            slice(h_start, h_end),
                            slice(w_start, w_end),
                            ))

                    overlap = tile.height - tile.height_overlap
                    offset = overlap - tile.height_offset % overlap
                    h_start += offset
                    h_end += offset

                if h_start < shape[3] and h_end > shape[3]:
                    self.append_tile((
                        slice(b, b + 1),
                        slice(0, 4),
                        slice(l_start, l_end),
                        slice(h_start, h_end),
                        slice(w_start, w_end),
                        ))

                overlap = tile.length - tile.length_overlap
                offset = overlap - tile.length_offset % overlap
                l_start += offset
                l_end += offset

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
