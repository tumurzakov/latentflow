import torch
import logging
import torch.nn.functional as F
from einops import rearrange

from .flow import Flow
from .tensor import Tensor
from .latent import Latent, NoisePredict
from .video import Video
from .mask import Mask

class Interpolate(Flow):
    def __init__(self,
            scale_factor=None,
            size=None,
            mode='nearest',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False,
            ):

        self.scale_factor = scale_factor
        self.size = size
        self.mode = mode
        self.align_corners = align_corners
        self.recompute_scale_factor = recompute_scale_factor
        self.antialias = antialias

    def apply(self, tensor) -> Tensor:
        tensor.onload()

        if isinstance(tensor, Tensor):
            t = tensor.tensor
        elif isinstance(tensor, Latent) or isinstance(tensor, NoisePredict):
            t = tensor.latent
        elif isinstance(tensor, mask):
            t = tensor.mask
        elif isinstance(tensor, Video):
            t = tensor.hwc()

        t = F.interpolate(
                t,
                scale_factor = self.scale_factor,
                size = self.size,
                mode=self.mode,
                align_corners=self.align_corners,
                recompute_scale_factor=self.recompute_scale_factor,
                antialias=self.antialias,
                )

        if isinstance(tensor, Video):
            result = Video('HWC', t)
        else:
            result = type(tensor)(t)

        del tensor
        result.offload()

        return result

class Resize(Interpolate):
    def __init__(self,
            scale_factor=None,
            size=None,
            mode='nearest',
            align_corners=None,
            recompute_scale_factor=None,
            antialias=False):

        super().__init__(scale_factor, size, mode, align_corners, recompute_scale_factor, antialias)

    def apply(self, tensor):
        t = None
        if isinstance(tensor, Tensor):
            t = tensor.tensor
        elif isinstance(tensor, Latent) or isinstance(tensor, NoisePredict):
            t = tensor.latent
            t = rearrange(t, 'b c f h w -> (b f) c h w')
            self.mode = 'nearest-exact'
        elif isinstance(tensor, Mask):
            t = tensor.mask
            t = rearrange(t, 'b c f h w -> (b f) c h w')
            self.mode = 'nearest-exact'
        elif isinstance(tensor, Video):
            t = tensor.hwc().to(torch.float)
            t = rearrange(t, 'b f h w c -> (b f) c h w')
            self.mode = 'bilinear'

        assert t is not None, 'Unknown type of input'

        t = super().apply(Tensor(t))
        t = t.tensor

        if isinstance(tensor, Latent) or isinstance(tensor, NoisePredict):
            t = rearrange(t, '(b f) c h w -> b c f h w', f=len(tensor))
            result = type(tensor)(t)
        if isinstance(tensor, Mask):
            t = rearrange(t, '(b f) c h w -> b c f h w', f=len(tensor))
            result = type(tensor)(t)
        elif isinstance(tensor, Video):
            t = rearrange(t, '(b f) c h w -> b f h w c', f=len(tensor))
            t = t.to(torch.uint8)
            result = type(tensor)('HWC', t)
        else:
            result = type(tensor)(t)

        return result
