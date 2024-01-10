import logging
import torch
from typing import Callable, List, Optional, Tuple, Union, Generator

from .latent import Latent
from .prompt_embeddings import PromptEmbeddings
from .unet import Unet
from .timesteps import Timesteps
from .flow import Flow

class Diffuse(Flow):

    def __init__(self,
            callback:Optional[Callable] = None,
            timesteps: Optional[Timesteps] = None,
            ):
        self.timesteps = timesteps
        self.current_timestep = 0
        self.callback = callback

        logging.debug("Diffuse init")

    def apply(self, state = None) -> Latent:

        timesteps = None
        try:
            timesteps = state['timesteps']
        except KeyError:
            pass

        if self.timesteps is None and timesteps is not None:
            self.timesteps = timesteps

        try:
            timestep_index, timestep = next(self.timesteps.generator)
            state['timestep_index'] = timestep_index
            state['timestep'] = timestep

            logging.debug("Diffuse apply %s", timestep)

            if self.callback is not None:
                latent = self.callback(
                        timestep=timestep,
                        state=state,
                    )

                state['latent'] = latent

            latent = self.apply(state)

        except StopIteration:
            logging.debug("Diffuse stop %s", state['latent'])

        return state['latent']


