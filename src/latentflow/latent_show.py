import logging
import mediapy
import numpy as np

from .flow import Flow
from .latent import Latent
from .vae_latent_decode import VaeLatentDecode
from .video_vae_latent_decode import VideoVaeLatentDecode
from .video_show import VideoShow

class LatentShow(Flow):
    def __init__(self,
            fps=25,
            vae = None,
            vae_batch=12,
            callback=None
            ):
        self.fps = fps
        self.vae_decode = VideoVaeLatentDecode(vae=vae, vae_batch=vae_batch)
        self.video_show = VideoShow(fps=fps)
        self.callback = callback

        logging.debug("LatentShow init %s %s",
                fps, vae_batch)

    def apply(self, latent) -> Latent:
        logging.debug('LatentShow(%s)', latent)

        l = latent
        if self.callback is not None:
            l = self.callback(latent)

        video = self.vae_decode.apply(l)
        self.video_show.apply(video)

        return latent

