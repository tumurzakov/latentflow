import torch
import logging
from typing import Callable, List, Optional, Tuple, Union, Generator

from .flow import Flow
from .latent import Latent
from .schedule import SchedulerInput
from .state import State

class AddNoise(Flow):
    def __init__(self,
            scheduler:Optional[SchedulerInput]=None,
            timestep = None,
            ):
        self.scheduler = scheduler
        self.timestep = timestep

        logging.debug('AddNoise init')


    def apply(self, latent) -> Latent:
        logging.debug('AddNoise apply %s %s', latent, self.timestep)

        noise = torch.randn_like(latent.latent)
        latent = self.scheduler.add_noise(latent.latent, noise, self.timestep)

        latent = Latent(latent=latent)
        logging.debug('AddNoise apply noised %s', latent)

        return latent

