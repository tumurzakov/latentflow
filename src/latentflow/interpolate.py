import torch
import logging
import torch.nn.functional as F

from .flow import Flow
from .tensor import Tensor

class Interpolate(Flow):
    def __init__(self,
            scale_factor=None,
            size=None,
            mode='nearest',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False):

        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def apply(self, tensor: Tensor) -> Tensor:
        tensor.onload()

        t = tensor.tensor
        t = F.interpolate(
                t,
                scale_factor = self.scale_factor,
                size = self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.recompute_scale_factor,
                antialias=self.antialias,
                )

        result = Tensor(t)

        tensor.offload()
        result.offload()

        return result

