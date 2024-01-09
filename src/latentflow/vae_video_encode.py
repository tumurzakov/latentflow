import torch
from einops import rearrange
import logging

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from .video_encode import VideoEncode
from .latent import Latent
from .video import Video


class VaeVideoEncode(VideoEncode):
    def __init__(self, vae: AutoencoderKL = None, vae_batch=1):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)

    @torch.no_grad()
    def apply(self, video: Video) -> Latent:
        logging.debug('VaeVideoEncode apply video %s', video)

        v = video.chw().to(self.vae.device)
        logging.debug('VaeVideoEncode apply v %s %s %s', v.shape, v.device, v.dtype)

        l = []
        for b in range(0, v.shape[0], self.vae_batch):
            l.append(self.encode(v[b:b+self.vae_batch]))
        l = torch.cat(l, dim=2)

        latent = Latent(latent=l, device=self.vae.device)
        logging.debug('VaeVideoEncode apply latent %s', latent)
        return latent

    def encode(self, video):
        weight_dtype = self.vae.dtype

        video = torch.stack([video])
        video = torch.tensor(video).to(self.vae.device, dtype=weight_dtype)
        pixel_values = (video / 127.5 - 1.0)

        video_length = video.shape[1]
        pixel_values = rearrange(pixel_values, "b f c h w -> (b f) c h w")

        latents = []
        for i in range(video_length):
            frame = torch.stack([pixel_values[i, :, :, :]])
            latents.append(self.vae.encode(frame).latent_dist.sample())
        latents = torch.cat(latents)

        latents = rearrange(latents, "(b f) c h w -> b c f h w", f=video_length)
        latents = latents * 0.18215

        return latents
