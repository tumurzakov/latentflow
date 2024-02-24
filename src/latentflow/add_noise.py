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
        self.timesteps = self.timesteps.to(self.onload_device)

    def offload(self):
        self.timesteps = self.timesteps.to(self.offload_device)

    def apply(self, latent) -> Latent:
        logging.debug('AddNoise apply %s', latent)

        self.onload()
        latent.onload()

        latent_timestep = self.timesteps.timesteps[:1]

        noise = torch.randn_like(latent.latent)
        latent = self.scheduler.add_noise(latent.latent, noise, latent_timestep)

        result = Latent(latent=latent)
        logging.debug('AddNoise apply noised %s', latent)

        latent.offload()
        result.offload()

        return latent

