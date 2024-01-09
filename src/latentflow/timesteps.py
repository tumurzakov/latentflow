import torch
import logging

from .flow import Flow

class Timesteps(Flow):
    def __init__(self,
            timesteps: torch.Tensor = None,
            num_inference_steps: int = 0,
            ):
        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps

        logging.debug(f'{self}')

        self.generator = self.generate()


    def __len__(self):
        return len(self.timesteps) if self.timesteps is not None else 0

    def __str__(self):
        return f'Timesteps({len(self)})'

    def __next__(self):
        return self.generator

    def generate(self):
        logging.debug("Timesteps iter %s", len(self))

        if len(self.timesteps) == 0:
            yield None

        for t in self.timesteps:
            yield t

