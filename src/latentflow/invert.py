import os
import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent
from .schedule import SchedulerInput
from .ddim_inversion import ddim_inversion

class Invert(Flow):
    def __init__(self,
            tokenizer=None,
            text_encoder=None,
            unet=None,
            scheduler:Optional[SchedulerInput]=None,
            steps: int = 10,
            prompt: str = "",
            video_length: int = None,
            temporal_context: int = None,
            cache: str = None,
            ):

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.steps = steps
        self.prompt = prompt
        self.video_length = video_length
        self.temporal_context = temporal_context
        self.cache = cache

        logging.debug("Invert init %s, %s, %s",
                type(self.scheduler), self.steps, self.prompt)

    def apply(self, latent: Latent) -> Latent:
        logging.debug("Invert apply %s", latent)

        if self.cache and os.path.isfile(self.cache):
            logging.debug("Invert load cache %s", self.cache)
            inv_latents = torch.load(self.cache)
        else:
            inv_latents = self.invert(latent.latent)

            if self.cache:
                logging.debug("Invert save cache %s", self.cache)
                torch.save(inv_latents, self.cache)

        latents = inv_latents[-1]
        latents = latents[:,:,:latent.latent.shape[2],:,:]
        latents = latents * self.scheduler.init_noise_sigma.to(self.text_encoder.device)
        return Latent(latent=latents)

    @torch.no_grad()
    def invert(self, latents: torch.Tensor) -> Latent:

        latents = latents.to(device=self.unet.device, dtype=self.unet.dtype)

        if self.video_length is None:
            self.video_length = latents.shape[2]

        if self.temporal_context is None:
            self.temporal_context = self.video_length

        inv_latents = [[] for s in range(self.steps+1)]
        for i in range(0, self.video_length, self.temporal_context):
            self.scheduler.set_timesteps(self.steps)
            batch_inv_latents = ddim_inversion(
                self.tokenizer,
                self.text_encoder,
                self.unet,
                self.scheduler,
                video_latent=latents[:,:,i:i+self.temporal_context,:,:],
                num_inv_steps=self.steps,
                prompt=self.prompt,
                desc='invert')

            for step, inv_latent in enumerate(batch_inv_latents):
                inv_latents[step].append(inv_latent)

        for s in range(self.steps+1):
            inv_latents[s] = torch.cat(inv_latents[s], dim=2)

        return inv_latents
