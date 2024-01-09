import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .latent import Latent
from .schedule import SchedulerInput
from .ddim_inversion import ddim_inversion

class Invert:
    def __init__(self,
            tokenizer=None,
            text_encoder=None,
            unet=None,
            scheduler:Optional[SchedulerInput]=None,
            steps: int = 10,
            prompt: str = "",
            video_length: int = None,
            temporal_context: int = None,
            ):

        self.tokenizer = tokenizer
        self.text_encoder = text_encoder
        self.unet = unet
        self.scheduler = scheduler
        self.steps = steps
        self.prompt = prompt
        self.video_length = video_length
        self.temporal_context = temporal_context

        logging.debug("Invert(%s, %s, %s)",
                type(self.scheduler), self.steps, self.prompt)

    def apply(self, latent: Latent) -> Latent:

        latents = latent.latent

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

        return Latent(latent=inv_latents[-1])
