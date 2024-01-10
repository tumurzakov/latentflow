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
        self._index = 0

    def __len__(self):
        return len(self.timesteps) if self.timesteps is not None else 0

    def __str__(self):
        return f'Timesteps({len(self)})'

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.timesteps):
            result = (self._index, self.timesteps[self._index])
            self._index += 1
            return result
        else:
            raise StopIteration
