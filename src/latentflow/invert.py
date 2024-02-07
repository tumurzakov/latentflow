import os
import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent
from .schedule import SchedulerInput
from .ddim_inversion import ddim_inversion
from .tile import Tile, TileGenerator

class Invert(Flow):
    def __init__(self,
            tokenizer=None,
            text_encoder=None,
            unet=None,
            scheduler:Optional[SchedulerInput]=None,
            steps: int = 10,
            prompt: str = "",
            video_length: int = None,
            tile_length: int = None,
            tile_height: int = None,
            tile_width: int = None,
            cache: str = None,
            ):

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.steps = steps
        self.prompt = prompt
        self.video_length = video_length
        self.tile_length = tile_length
        self.tile_height = tile_height
        self.tile_width = tile_width
        self.cache = cache

        logging.debug("Invert init %s, %s, %s",
                type(self.scheduler), self.steps, self.prompt)

    def apply(self, latent: Latent) -> Latent:
        logging.debug("Invert apply %s", latent)

        if self.cache and os.path.isfile(self.cache):
            logging.debug("Invert load cache %s", self.cache)
            inv_latents = torch.load(self.cache)
        else:
            tile_generator = TileGenerator(
                    Tile(
                        length=self.tile_length,
                        height=self.tile_height,
                        width=self.tile_width,
                        ),
                    latent,
                    do_classifier_free_guidance=False,
                    )

            inv_latents = torch.zeros_like(latent.latent)
            for tile_index, tile in enumerate(tile_generator.tiles):
                latents = latent.latent[tile]
                inv_latents[tile] = self.invert(
                    latents,
                    f'{tile_index+1}/{len(tile_generator.tiles)}',
                    )[-1]

            if self.cache:
                logging.debug("Invert save cache %s", self.cache)
                torch.save(inv_latents, self.cache)

        latents = inv_latents
        latents = latents[:,:,:latent.latent.shape[2],:,:]
        latents = latents * self.scheduler.init_noise_sigma
        return Latent(latent=latents)

    @torch.no_grad()
    def invert(self, latents: torch.Tensor, desc = '') -> Latent:

        latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        if self.video_length is None:
            self.video_length = latents.shape[2]

        if self.tile_length is None:
            self.tile_length = self.video_length

        inv_latents = [[] for s in range(self.steps+1)]
        for i in range(0, self.video_length, self.tile_length):
            self.scheduler.set_timesteps(self.steps)
            batch_inv_latents = ddim_inversion(
                self.tokenizer,
                self.text_encoder,
                self.unet,
                self.scheduler,
                video_latent=latents[:,:,i:i+self.tile_length,:,:],
                num_inv_steps=self.steps,
                prompt=self.prompt,
                desc=f'invert {desc}')

            for step, inv_latent in enumerate(batch_inv_latents):
                inv_latents[step].append(inv_latent)

        for s in range(self.steps+1):
            inv_latents[s] = torch.cat(inv_latents[s], dim=2)

        return inv_latents
