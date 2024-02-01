import torch
import logging

from .interpolate import Interpolate
from .latent import Latent
from .tensor import Tensor

class LatentInterpolate(Interpolate):
    def __init__(self, scale_factor, mode='trilinear'):
        super().__init__(scale_factor = scale_factor, mode = mode)

    def apply(self, latent):
        l = super().apply(Tensor(latent.latent))
        return Latent(l.tensor)
