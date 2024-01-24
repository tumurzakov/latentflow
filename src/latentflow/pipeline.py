import torch
import logging

from .flow import Flow
from .latent import Latent

class Pipeline(Flow):
    def __init__(self, pipe, callback = None, **kwargs):
        self.pipe = pipe
        self.callback = callback
        self.kwargs = kwargs

    def apply(self, latent):
        kwargs = self.kwargs

        logging.debug("Pipeline %s %s",
                latent, type(self.pipe))

        if isinstance(latent, Latent) and 'latents' not in kwargs:
            kwargs['latents'] = latent.latent.to(
                    device=self.pipe.unet.device,
                    dtype=self.pipe.unet.dtype)

        kwargs['output_type'] = 'latent'

        result = self.pipe(**kwargs)

        if isinstance(result, tuple):
            result = result[0]

        if self.callback is not None:
            result=self.callback(result)

        return Latent(result)
