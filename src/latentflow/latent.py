import logging
import torch
from typing import List, Optional, Tuple, Union, Generator
import torch.nn.functional as F
from .flow import Flow
from .tensor import Tensor

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
            device = latent.device

        self.shape = shape
        self.device = device

        if self.latent is None:
            if full is not None:
                self.latent = torch.full(self.shape, full)
            elif self.shape is not None:
                self.latent = torch.randn(self.shape)

        if device is not None:
            self.latent = self.latent.to(device)

        logging.debug("Latent init %s %s", type(self), self)

    def save(self, path):
        torch.save(self.latent, path)
        return self

    def load(self, path, device='cpu'):
        return Latent(torch.load(path).to(device))

    def clone(self):
        return type(self)(self.latent.clone())

    def apply(self, value):
        logging.debug("Latent apply %s %s", self, value)
        self.set(value)
        return self

    def resize(self, size):
        l = F.interpolate(
                self.latent,
                size=size,
                mode='trilinear',
                align_corners=False
                )
        return Latent(l)

    def set(self, value):
        assert isinstance(value, Latent) \
            or isinstance(value, Tensor) \
            or isinstance(value, torch.Tensor) \
            , f"Should be latent {type(value)}"

        logging.debug("Latent set %s %s", self, value)

        tensor = value
        if isinstance(value, Latent):
            tensor = value.latent
        elif isinstance(value, Tensor):
            tensor = value.tensor

        if self.latent is None:
            self.latent = tensor
        else:
            self.latent[:] = tensor


    def __str__(self):
        if self.latent is not None:
            shape = self.latent.shape
            device = self.latent.device
            dtype = self.latent.dtype
            return f'Latent({shape}, {device}, {dtype})'

        return f'Latent(None)'

    def __getitem__(self, key):
        return Latent(self.latent[key])


class NoisePredict(Latent):
    pass

class LatentAdd(Flow):
    def __init__(self, latent, key=None):
        self.latent = latent
        self.key = key

    def apply(self, other):
        logging.debug("LatentAdd.apply %s[%s] %s", self.latent, self.key, other)

        s = self.latent.latent
        l = other.latent

        if self.key is not None:
            s[self.key] += l
        else:
            s += l

        return self.latent
