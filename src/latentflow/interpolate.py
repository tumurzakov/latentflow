import torch
import logging
import torch.nn.functional as F

from .flow import Flow

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

    def apply(self, tensor: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
                tensor,
                scale_factor = self.scale_factor,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.recompute_scale_factor,
                antialias=self.antialias,
                )

