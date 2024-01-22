import torch
import logging
from einops import rearrange
from sgm.util import instantiate_from_config, load_model_from_config
from omegaconf import OmegaConf
import torch
import gc
from safetensors.torch import load_file as load_safetensors
import math
from tqdm import tqdm

from .latent_decode import LatentDecode
from .video import Video
from .latent import Latent

class VideoVaeLatentDecode(LatentDecode):
    def __init__(self, vae = None, vae_batch=12):
        self.vae = vae
        self.vae_batch = vae_batch

        logging.debug('VaeVideoEncode init %d', vae_batch)

    @torch.no_grad()
    def apply(self, latent: Latent) -> Video:
        logging.debug('VaeVideoEncode apply latent %s', latent)

        v = self.decode(latent.latent)
        v = rearrange(v, 'b c f h w -> b f h w c')
        video = Video('HWC', video=v)

        logging.debug('VaeVideoEncode apply video %s', video)
        return video

    def decode(self, latents):
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype)
        return self.vae.decode(latents, self.vae_batch)
