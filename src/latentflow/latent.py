import logging
import torch
from typing import List, Optional, Tuple, Union, Generator
from .flow import Flow
from .mask import Mask

class Latent(Flow):
    def __init__(self,
            latent: torch.Tensor=None,
            shape: tuple = None,
            device: Optional[Union[str, torch.device]] = None,
            full = None,
            ):

        self.latent = None

        if latent is not None:
            assert len(latent.shape) == 5
            self.latent = latent
            shape = latent.shape

        self.shape = shape

        if self.latent is None:
            if full is not None:
                self.latent = torch.full(self.shape, full)
            else:
                self.latent = torch.randn(self.shape)

        if device is not None:
            self.latent = self.latent.to(device)

        logging.debug("Latent init %s %s", type(self), self)

    def apply(self, value):
        assert isinstance(value, Latent), "Should be latent"

        logging.debug("Latent apply %s %s", self, value)

        if self.latent is None:
            self.latent = value.latent
        else:
            self.latent[:] = value.latent

        return self

    def set(self, value):
        assert isinstance(value, Latent), "Should be latent"

        logging.debug("Latent set %s %s", self, value)

        if self.latent is None:
            self.latent = value.latent
        else:
            self.latent[:] = value.latent


    def __str__(self):
        if self.latent is not None:
            shape = self.latent.shape
            device = self.latent.device
            dtype = self.latent.dtype
            return f'Latent({shape}, {device}, {dtype})'

        return f'Latent(None)'

class NoisePredict(Latent):
    pass

class LatentMaskMerge(Flow):
    def __init__(self, latent: Latent, mask: Mask = None):
        self.latent = latent

        self.mask = mask
        if self.mask is None:
            self.mask = Mask(torch.ones_like(self.latent.latent))

    def apply(self, latent: Latent) -> Latent:
        return Latent(self.latent.latent * (1 - self.mask.mask) + latent.latent * self.mask.mask)
