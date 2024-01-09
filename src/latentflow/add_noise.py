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
            device: Optional[Union[str, torch.device]] = None,
            ):
        self.scheduler = scheduler
        self.device = device

        logging.debug('AddNoise init %s', self.device)

    def apply(self, state: State) -> Latent:
        latent = state['latent']
        timesteps = state['timesteps']

        logging.debug('AddNoise apply %s %s', latent, timesteps)

        noise = torch.randn_like(latent.latent)
        timestep = timesteps.timesteps[:1]
        latent = self.scheduler.add_noise(latent.latent, noise, timestep)

        latent = Latent(latent=latent)
        logging.debug('AddNoise apply noised %s', latent)

        return latent

