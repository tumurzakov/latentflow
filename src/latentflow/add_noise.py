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
            timesteps = None,
            device: Optional[Union[str, torch.device]] = 'cuda',
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):

        self.scheduler = scheduler
        self.timesteps = timesteps
        self.onload_device = onload_device

        logging.debug('AddNoise init')

    def onload(self):
        self.timesteps.onload()

    def offload(self):
        self.timesteps.offload()

    def apply(self, latent) -> Latent:
        logging.debug('AddNoise apply %s', latent)

        self.onload()
        latent.onload()

        latent_timestep = self.timesteps.timesteps[:1]

        noise = torch.randn_like(latent.latent)
        noised_latent = self.scheduler.add_noise(latent.latent, noise, latent_timestep)

        result = Latent(latent=noised_latent)
        logging.debug('AddNoise apply noised %s', result)

        latent.offload()
        result.offload()

        return latent

