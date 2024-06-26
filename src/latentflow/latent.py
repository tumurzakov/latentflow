import os
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
            onload_device: str='cuda',
            offload_device: str='cpu',
            full = None,
            ):

        self.latent = None

        if latent is not None:
            self.latent = latent
            shape = latent.shape

        self.shape = shape
        self.onload_device = onload_device
        self.offload_device = offload_device

        if self.latent is None:
            if full is not None:
                self.latent = torch.full(self.shape, full)
            elif self.shape is not None:
                self.latent = torch.randn(self.shape)

        logging.debug("Latent init %s %s", type(self), self)

    def onload(self):
        self.latent = self.latent.to(self.onload_device)

    def offload(self):
        self.latent = self.latent.to(self.offload_device)

    def save(self, output_path):
        dirname = os.path.dirname(output_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        torch.save(self.latent, output_path)

        return self

    def load(self, path=None):
        if isinstance(self, str) and path is None:
            path = self

        return Latent(torch.load(path))

    def clone(self):
        return type(self)(self.latent.clone())

    def apply(self, value):
        logging.debug("Latent apply %s %s", self, value)
        self.onload()
        self.set(value)
        self.offload()
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

    def __len__(self):
        return self.latent.shape[2]

    def zeros_like(self):
        return Latent(torch.zeros_like(self.latent))


class NoisePredict(Latent):
    def __str__(self):
        if self.latent is not None:
            shape = self.latent.shape
            device = self.latent.device
            dtype = self.latent.dtype
            return f'NoisePredict({shape}, {device}, {dtype})'


class LatentAdd(Flow):
    def __init__(self, latent, key=None, mask=None):
        self.latent = latent
        self.key = key
        self.mask = mask

    def apply(self, other):
        logging.debug("LatentAdd.apply %s[%s] %s %s", \
                self.latent, self.key, other, self.mask)

        other.onload()
        self.latent.onload()

        if self.mask is not None:
            self.mask.onload()

        s = self.latent.latent
        l = other.latent

        if self.mask is not None:
            if self.key is not None:
                l = l * self.mask[self.key].mask
            else:
                l = l * self.mask.mask

        if self.key is not None:
            s[self.key] += l
        else:
            s += l

        other.offload()
        self.latent.offload()

        if self.mask is not None:
            self.mask.offload()

        return self.latent

class LatentSet(Flow):
    def __init__(self, latent, key=None, mask=None):
        self.latent = latent
        self.key = key
        self.mask = mask

    def apply(self, other):
        logging.debug("LatentSet.apply %s[%s] %s %s", \
                self.latent, self.key, other, self.mask)

        other.onload()
        self.latent.onload()

        if self.mask is not None:
            self.mask.onload()

        s = self.latent.latent
        l = other.latent

        if self.mask is not None:
            if self.key is not None:
                s[self.key] = s[self.key] * self.mask.invert()[self.key].mask
                l = l * self.mask[self.key].mask
            else:
                s = s * self.mask.invert().mask
                l = l * self.mask.mask

        if self.key is not None:
            s[self.key] += l
        else:
            s += l

        other.offload()
        self.latent.offload()

        if self.mask is not None:
            self.mask.offload()

        return self.latent
