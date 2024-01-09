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
            device: Optional[Union[str, torch.device]] = None,
            ):
        self.scheduler = scheduler
        self.device = device

        logging.debug('Noise init %s', self.device)

    def apply(self, latent: Latent) -> Latent:
        latents = latent.latent
        latents = torch.randn_like(latents).to(self.device)
        latents = latents * self.scheduler.init_noise_sigma
        latent.latent = latents

        logging.debug('Noise apply noised %s', latent)

        return latent

