import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent
from .schedule import SchedulerInput
from .state import State

class Noise(Flow):
    def __init__(self,
            scheduler:Optional[SchedulerInput]=None,
            scale: Optional[int] = 1.0,
            ):
        self.scheduler = scheduler
        self.scale = scale

        logging.debug('Noise init %s', scale)

    def apply(self, latent: Latent) -> Latent:
        latents = latent.latent
        noise = torch.randn_like(latents).to(latents.device)
        noise = noise * self.scheduler.init_noise_sigma

        latent.latent = latents * (1.0-self.scale) + noise * self.scale

        logging.debug('Noise apply noised %s', latent)

        return latent

