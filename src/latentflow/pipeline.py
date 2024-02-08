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

        if 'output_type' not in kwargs:
            kwargs['output_type'] = 'latent'

        if hasattr('__orig_call__', self.pipe):
            result = self.pipe.__orig_call__(**kwargs)
        else:
            result = self.pipe(**kwargs)

        assert result is not None, "pipeline result should not be none"

        if self.callback is not None:
            result=self.callback(result)

        elif isinstance(result, tuple):
            result = result[0]

        logging.debug("Pipeline result %s", result.shape)

        return Latent(result)
