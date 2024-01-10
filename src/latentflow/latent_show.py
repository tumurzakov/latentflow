import logging
import mediapy
import numpy as np
from diffusers import AutoencoderKL

from .flow import Flow
from .latent import Latent
from .vae_latent_decode import VaeLatentDecode
from .video_show import VideoShow

class LatentShow(Flow):
    def __init__(self, fps=25, vae: AutoencoderKL = None, vae_batch=1):
        self.fps = fps
        self.vae_decode = VaeLatentDecode(vae=vae, vae_batch=vae_batch)
        self.video_show = VideoShow(fps=fps)
        logging.debug("LatentShow init %s %s %s %s",
                fps, vae.device, vae.dtype, vae_batch)

    def apply(self, latent: Latent) -> Latent:
        logging.debug('LatentShow(%s)', latent)

        video = self.vae_decode.apply(latent)
        self.video_show.apply(video)

        return latent

