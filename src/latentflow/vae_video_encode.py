import os
import torch
from einops import rearrange
import logging
from tqdm import tqdm

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
            onload_device='cuda',
            offload_device='cpu',
            ):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch
        self.cache = cache
        self.video_length = video_length
        self.start_frame = start_frame
        self.onload_device = onload_device
        self.offload_device = offload_device

        if self.cache is not None:
            cache_dir = os.path.dirname(self.cache)

            if not os.path.isdir(cache_dir):
                os.mkdir(cache_dir)

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)
        logging.debug('VaeVideoEncode video_length %s %s',
                video_length, start_frame)

    def onload(self):
        self.vae = self.vae.to(self.onload_device)

    def offload(self):
        self.vae = self.vae.to(self.offload_device)

    @torch.no_grad()
    def apply(self, video: Video) -> Latent:
        logging.debug('VaeVideoEncode apply video %s', video)

        video.onload()
        self.onload()

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
                self.video_length = len(video)

            latents = []
            for v in videos:
                latent = []
                for b in tqdm(range(self.start_frame, self.video_length, self.vae_batch), desc='vae'):
                    latent.append(self.encode(v[b:b+self.vae_batch]))
                latent = torch.cat(latent, dim=2)
                latents.append(latent)

            latents = torch.cat(latents, dim=0)

            if self.cache:
                logging.debug('VaeVideoEncode save cache latent %s', self.cache)
                torch.save(latents, self.cache)

        latent = Latent(latent=latents)
        logging.debug('VaeVideoEncode apply latent %s', latent)

        latent.offload()
        self.offload()

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
        latents = latents * self.vae.config.scaling_factor

        return latents
