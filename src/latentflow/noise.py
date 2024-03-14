import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent
from .schedule import SchedulerInput
from .state import State
from .mask import Mask

class Noise(Flow):
    def __init__(self,
            scheduler:Optional[SchedulerInput]=None,
            scale: Optional[Union[int, Callable]] = 1.0,
            mask: Mask = None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):
        self.scheduler = scheduler
        self.scale = scale if scale is not None else 0.0
        self.mask = mask
        self.onload_device = onload_device
        self.offload_device = offload_device


        logging.debug('Noise init %s', scale)

    def onload(self):
        if self.mask is not None:
            self.mask.onload()

    def offload(self):
        if self.mask is not None:
            self.mask.offload()

    def apply(self, latent: Latent) -> Latent:

        self.onload()
        latent.onload()

        latents = latent.latent
        noise = torch.randn_like(latents).to(latents.device)
        noise = noise * self.scheduler.init_noise_sigma

        scale = self.scale
        if callable(self.scale):
            scale = torch.tensor([self.scale(i) for i in range(latents.shape[2])]).to(latents.device)
            scale = scale.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)

        noise_latent = latents * (1.0-scale) + noise * scale

        if self.mask is not None:
            noise_latent = latents * (1.0 - self.mask.mask) + noise_latent * self.mask.mask

        output = Latent(noise_latent)

        logging.debug('Noise apply %s', output)

        latent.offload()
        output.offload()
        self.offload()

        return output
