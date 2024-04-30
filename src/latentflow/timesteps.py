import torch
import logging

from .flow import Flow

class Timesteps(Flow):
    def __init__(self,
            timesteps: torch.Tensor = None,
            num_inference_steps: int = 0,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):

        self.timesteps = timesteps
        self.num_inference_steps = num_inference_steps
        self.onload_device = onload_device
        self.offload_device = offload_device

        logging.debug(f'{self}')
        self._index = 0

    def onload(self):
        self.timesteps = self.timesteps.to(self.onload_device)

    def offload(self):
        self.timesteps = self.timesteps.to(self.offload_device)

    def set(self, timesteps):
        self.timesteps = timesteps.timesteps
        return self

    def __len__(self):
        return len(self.timesteps) if self.timesteps is not None else 0

    def __str__(self):
        return f'Timesteps({len(self)})'

    def __getitem__(self, key):
        return self.timesteps[key]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.timesteps):
            result = self.timesteps[self._index]
            self._index += 1
            return result
        else:
            raise StopIteration
