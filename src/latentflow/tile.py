import torch
import logging
import random

from .flow import Flow
from .latent import Latent
from .tensor import Tensor
from .prompt_embeddings import PromptEmbeddings

import numpy as np

def ordered_halving(i):
    return int('{:064b}'.format(i)[::-1], 2) / (1 << 64)


def uniform(step, n, context_size, strides, overlap, closed_loop=True):
    if n <= context_size:
        yield list(range(n))
        return
    strides = min(strides, int(np.ceil(np.log2(n / context_size))) + 1)
    for stride in 1 << np.arange(strides):
        pad = int(round(n * ordered_halving(step)))
        for j in range(
                int(ordered_halving(step) * stride) + pad,
                n + pad + (0 if closed_loop else -overlap),
                (context_size * stride - overlap)
        ):
            yield [e % n for e in range(j, j + context_size * stride, stride)]

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

class UniformFrameTileGenerator(Flow):
    def __init__(self,
            timestep_index,
            context_size,
            strides,
            overlap,
            latent,
            pixel_infer_count=None,
            do_classifier_free_guidance=True,
            ):
        self.timestep_index = timestep_index
        self.context_size = context_size
        self.strides = strides
        self.overlap = overlap
        self.latent = latent
        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.pixel_infer_count = pixel_infer_count
        if self.pixel_infer_count is None:
            self.pixel_infer_count = torch.zeros_like(latent.latent)
        else:
            assert isinstance(self.pixel_infer_count, Tensor), "pixel_infer_count should be Tensor"
            self.pixel_infer_count = self.pixel_infer_count.tensor

        self.tiles = []
        self.uniform()

        self._index = 0
        logging.debug("UniformFrameTileGenerator init [%s] tiles", len(self.tiles))

    def append_tile(self, tile):
        if self.do_classifier_free_guidance:
            bslice = slice(tile[0].start*2, tile[0].stop*2)
            tile = (
                    bslice,
                    tile[1],
                    tile[2],
                    tile[3],
                    tile[4],
                    )

        self.tiles.append(tile)
        self.pixel_infer_count[tile] += 1

    def uniform(self):
        for seq in uniform(
                self.timestep_index,
                self.latent.shape[2],
                self.context_size,
                self.strides,
                self.overlap):

            self.append_tile((
                slice(0, self.latent.shape[0]),
                slice(0, self.latent.shape[1]),
                seq,
                slice(0, self.latent.shape[3]),
                slice(0, self.latent.shape[4]),
                ))

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
            logging.debug("UniformFrameGenerator next %s %s", self._index, tile)
            self._index += 1

            return tile
        else:
            raise StopIteration

class TileGenerator(Flow):
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

        self.do_classifier_free_guidance = do_classifier_free_guidance

        self.pixel_infer_count = pixel_infer_count
        if self.pixel_infer_count is None:
            self.pixel_infer_count = torch.zeros_like(latent.latent)
        else:
            assert isinstance(self.pixel_infer_count, Tensor), "pixel_infer_count should be Tensor"
            self.pixel_infer_count = self.pixel_infer_count.tensor

        self.tiles = []
        self.audit = torch.zeros_like(latent.latent).int()

        self.generate()

        self._index = 0

        self.cover = self.audit.sum() / self.audit.numel()
        logging.debug("TileGenerator tiles [%s] cover=%s",
                len(self.tiles), self.cover.item())

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

    def append_tile(self, tile):
        if self.audit[tile].sum() == self.audit[tile].numel():
            return

        if self.do_classifier_free_guidance:
            bslice = slice(tile[0].start*2, tile[0].stop*2)
            tile = (
                    bslice,
                    tile[1],
                    tile[2],
                    tile[3],
                    tile[4],
                    )

        self.tiles.append(tile)
        self.audit[tile] = 1
        self.pixel_infer_count[tile] += 1

    def check_tile(self, t, s):
        ok = True
        for i, k in enumerate(t):
            if isinstance(k, slice):
                ok = ok and (
                        k.start != k.stop
                        and k.start >= 0
                        and k.stop <= s[i]
                        )
        return ok

    def generate(self):
        s = self.latent.latent.shape
        t = self.tile

        for l in range(-t.length, s[2]+t.length, t.length-t.length_overlap):
            for h in range(-t.height, s[3]+t.height, t.height-t.height_overlap):
                for w in range(-t.width, s[4]+t.width, t.width-t.height_overlap):

                    l_start = l + t.length_offset
                    l_end = l + t.length + t.length_offset
                    l_list = list([x%s[2] for x in range(l_start, l_end)])
                    l_list = [ s[2]+x if x<0 else x for x in l_list ]

                    h_start = h + t.height_offset
                    h_end = h + t.height + t.height_offset
                    if h_start < 0:
                        h_start = 0
                    if h_end > s[3]:
                        h_end = s[3]

                    w_start = w + t.width_offset
                    w_end = w + t.width + t.width_offset
                    if w_start < 0:
                        w_start = 0
                    if w_end > s[4]:
                        w_end = s[4]

                    tile = (
                        slice(0,s[0]),
                        slice(0,s[1]),
                        l_list,
                        slice(h_start, h_end),
                        slice(w_start, w_end),
                    )

                    if self.check_tile(tile, s):
                        self.append_tile(tile)

class AddTileEncoding(Flow):
    def __init__(self,
            tile,
            tokenizer,
            text_encoder,
            onload_device='cuda',
            offload_device='cpu'):

        self.tile = tile
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.text_encoder.to(self.onload_device)

    def offload(self):
        self.text_encoder.to(self.offload_device)

    def apply(self, embeddings):
        self.onload()
        embeddings.onload()

        embeds = embeddings.embeddings

        uncond, cond = embeds.chunk(2)

        tile = self.tile
        tile_z, tile_y, tile_x, tile_h, tile_w = tile[2][0],tile[3].start,tile[4].start,tile[3].stop,tile[4].stop

        prompt = f"tileZ{tile_z}X{tile_y}Y{tile_x}W{tile_h}H{tile_w}"
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        tile_cond = self.text_encoder(prompt_ids.to(self.onload_device))[0]
        video_length = cond.shape[0]
        tile_cond = tile_cond.repeat(video_length, 1, 1)
        tile_uncond = torch.zeros_like(tile_cond)

        cond = torch.cat([cond, tile_cond], dim=1)
        uncond = torch.cat([uncond, tile_uncond], dim=1)
        embeds = torch.cat([uncond, cond])
        result = PromptEmbeddings(embeds)

        embeddings.offload()
        result.offload()
        self.offload()

        return result

class AddFrameEncoding(Flow):
    def __init__(self,
            frame,
            tokenizer,
            text_encoder,
            onload_device='cuda',
            offload_device='cpu'):

        self.frame = frame
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.text_encoder.to(self.onload_device)

    def offload(self):
        self.text_encoder.to(self.offload_device)

    def apply(self, embeddings):
        self.onload()
        embeddings.onload()

        embeds = embeddings.embeddings

        uncond, cond = embeds.chunk(2)

        prompt = f"frame{self.frame}"
        prompt_ids = self.tokenizer(
            prompt,
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        ).input_ids
        frame_cond = self.text_encoder(prompt_ids.to(self.onload_device))[0]
        video_length = cond.shape[0]
        frame_cond = frame_cond.repeat(video_length, 1, 1)
        frame_uncond = torch.zeros_like(frame_cond)

        cond = torch.cat([cond, frame_cond], dim=1)
        uncond = torch.cat([uncond, frame_uncond], dim=1)
        embeds = torch.cat([uncond, cond])
        result = PromptEmbeddings(embeds)

        embeddings.offload()
        result.offload()
        self.offload()

        return result
