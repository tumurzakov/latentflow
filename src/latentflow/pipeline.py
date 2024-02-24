import torch
import logging

from .flow import Flow
from .latent import Latent

class Pipeline(Flow):
    def __init__(self,
            pipe,
            callback = None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            **kwargs):
        self.pipe = pipe
        self.callback = callback
        self.kwargs = kwargs
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.pipe = self.pipe.to(self.onload_device)

    def offload(self):
        self.pipe = self.pipe.to(self.offload_device)

    def apply(self, latent):
        kwargs = self.kwargs

        self.onload()

        logging.debug("Pipeline %s %s",
                latent, type(self.pipe))

        if isinstance(latent, Latent) and 'latents' not in kwargs and latent.latent is not None:
            kwargs['latents'] = latent.latent.to(
                    device=self.pipe.unet.device,
                    dtype=self.pipe.unet.dtype)

        if 'output_type' not in kwargs:
            kwargs['output_type'] = 'latent'

        if hasattr(self.pipe, '__orig_call__'):
            result = self.pipe.__orig_call__(**kwargs)
        else:
            result = self.pipe(**kwargs)

        assert result is not None, "pipeline result should not be none"

        if hasattr(result, 'images'):
            result = result.images

        if self.callback is not None:
            result=self.callback(result)

        elif isinstance(result, tuple):
            result = result[0]

        result = Latent(result)

        logging.debug("Pipeline result %s", result)

        self.offload()
        result.offload()

        return result
