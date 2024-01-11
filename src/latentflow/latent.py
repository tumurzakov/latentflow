import logging
import torch
from typing import List, Optional, Tuple, Union, Generator
from .flow import Flow

class Latent(Flow):
    def __init__(self,
            latent: torch.Tensor=None,
            batches: int = 1,
            channels: int = 4,
            width: int = None,
            height: int = None,
            length: int = None,
            device: Optional[Union[str, torch.device]] = None,
            ):

        self.latent = None

        if latent is not None:
            assert len(latent.shape) == 5
            batches, channels, length, height, width = latent.shape
            self.latent = latent

        self.batches = batches
        self.channels = channels
        self.width = width
        self.height = height
        self.length = length

        if self.latent is None:
            self.latent = torch.randn((self.batches,self.channels,self.length, self.height, self.width))

        if device is not None:
            self.latent = self.latent.to(device)

        logging.debug("Latent init %s", self)

    def set(self, value):
        assert isinstance(value, Latent), "Should be latent"

        logging.debug("Latent set %s %s", self, value)

        self.latent[:] = value.latent

    def __str__(self):
        if self.latent is not None:
            shape = self.latent.shape
            device = self.latent.device
            dtype = self.latent.dtype
            return f'Latent({shape}, {device}, {dtype})'

        return f'Latent(None)'

