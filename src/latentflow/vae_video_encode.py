import os
import torch
from einops import rearrange
import logging

from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from .video_encode import VideoEncode
from .latent import Latent
from .video import Video


class VaeVideoEncode(VideoEncode):
    def __init__(self,
            vae: AutoencoderKL = None,
            vae_batch:int = 1,
            cache: str = None,
            video_length: int = None,
            start_frame: int = 0,
            ):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch
        self.cache = cache
        self.video_length = video_length
        self.start_frame = start_frame

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)
        logging.debug('VaeVideoEncode video_length %s %s',
                video_length, start_frame)

    @torch.no_grad()
    def apply(self, video: Video) -> Latent:
        logging.debug('VaeVideoEncode apply video %s', video)

        if self.cache and os.path.isfile(self.cache):
            logging.debug('VaeVideoEncode load cache latent %s', self.cache)
            latents = torch.load(self.cache)

            if self.video_length is not None:
                latents = latents[:,:,self.start_frame:self.video_length,:,:]
        else:
            videos = video.chw().to(self.vae.device)
            logging.debug('VaeVideoEncode apply videos %s %s %s',
                    videos.shape, videos.device, videos.dtype)

            if self.video_length is None:
                self.video_length = v.shape[1]

            latents = []
            for v in videos:
                latent = []
                for b in range(self.start_frame, self.video_length, self.vae_batch):
                    latent.append(self.encode(v[b:b+self.vae_batch]))
                latent = torch.cat(latent, dim=2)
                latents.append(latent)

            latents = torch.cat(latents, dim=0)

            if self.cache:
                logging.debug('VaeVideoEncode save cache latent %s', self.cache)
                torch.save(latents, self.cache)

        latent = Latent(latent=latents, device=self.vae.device)
        logging.debug('VaeVideoEncode apply latent %s', latent)

        return latent

    @torch.no_grad()
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
