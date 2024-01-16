import torch
import logging
from einops import rearrange

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from .latent_decode import LatentDecode
from .video import Video
from .latent import Latent

class VaeLatentDecode(LatentDecode):
    def __init__(self, vae: AutoencoderKL = None, vae_batch=1):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)


    @torch.no_grad()
    def apply(self, latent: Latent) -> Video:
        logging.debug('VaeVideoEncode apply latent %s', latent)

        v = self.decode(latent.latent)
        video = Video('HWC', video=v, device=self.vae.device)

        logging.debug('VaeVideoEncode apply video %s', video)
        return video

    def decode(self, latents):
        video_length = latents.shape[2]
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = 1 / 0.18215 * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        video = []
        for frame_idx in range(0, latents.shape[0], self.vae_batch):
            video.append(self.vae.decode(
                latents[frame_idx:frame_idx+self.vae_batch]
                ).sample)

        video = torch.cat(video)
        video = rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video = (video / 2 + 0.5).clamp(0, 1)
        video = video.float()

        return video
