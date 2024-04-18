import logging
import mediapy
import numpy as np

from .flow import Flow
from .latent import Latent
from .vae_latent_decode import VaeLatentDecode
from .video_show import VideoShow

class LatentShow(Flow):
    def __init__(self,
            fps=25,
            vae = None,
            vae_batch=12,
            callback=None,
            onload_device='cuda',
            offload_device='cpu',
            progress_bar=True,
            ):
        self.fps = fps
        self.vae = vae
        self.vae_batch = vae_batch
        self.callback = callback
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.progress_bar = progress_bar

        logging.debug("LatentShow init %s %s",
                fps, vae_batch)

    def onload(self):
        self.vae = self.vae.to(self.onload_device)

    def offload(self):
        self.vae = self.vae.to(self.offload_device)

    def apply(self, latent) -> Latent:
        logging.debug('LatentShow(%s)', latent)

        self.onload()
        latent.onload()

        self.vae_decode = VaeLatentDecode(vae=self.vae, vae_batch=self.vae_batch, progress_bar=self.progress_bar)
        self.video_show = VideoShow(fps=self.fps)

        l = latent
        if self.callback is not None:
            l = self.callback(latent)

        video = self.vae_decode.apply(l)
        self.video_show.apply(video)

        latent.offload()
        self.offload()

        return latent

