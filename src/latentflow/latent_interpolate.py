import torch
import logging

from .interpolate import Interpolate
from .latent import Latent
from .tensor import Tensor

class LatentInterpolate(Interpolate):
    def __init__(self, scale_factor, mode='trilinear'):
        super().__init__(scale_factor = scale_factor, mode = mode)

    def apply(self, latent):
        latent.onload()
        l = super().apply(Tensor(latent.latent))
        result = Latent(l.tensor)

        latent.offload()
        result.offload()

        return result
