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
            scale: Optional[int] = 1.0,
            mask: Mask = None,
            ):
        self.scheduler = scheduler
        self.scale = scale
        self.mask = mask

        logging.debug('Noise init %s', scale)

    def apply(self, latent: Latent) -> Latent:
        latents = latent.latent
        noise = torch.randn_like(latents).to(latents.device)
        noise = noise * self.scheduler.init_noise_sigma

        noise_latent = latents * (1.0-self.scale) + noise * self.scale

        if self.mask is not None:
            noise_latent = latents * (1.0 - self.mask.mask) + noise_latent * self.mask.mask

        output = Latent(noise_latent)

        logging.debug('Noise apply %s', output)

        return output
