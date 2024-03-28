import torch
import logging
from einops import rearrange

from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder
from diffusers.image_processor import VaeImageProcessor
from tqdm import tqdm

from .latent_decode import LatentDecode
from .video import Video
from .latent import Latent

class VaeLatentDecode(LatentDecode):
    def __init__(self,
            vae: AutoencoderKL = None,
            vae_batch=1,
            onload_device='cuda',
            offload_device='cpu',
            ):
        self.vae = vae
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.vae_batch = vae_batch
        self.onload_device = onload_device
        self.offload_device = offload_device

        logging.debug('VaeVideoEncode init %d, vae(%s,%s)',
                vae_batch,
                vae.device,
                vae.dtype)

    def onload(self):
        self.vae = self.vae.to(self.onload_device)

    def offload(self):
        self.vae = self.vae.to(self.offload_device)


    @torch.no_grad()
    def apply(self, latent: Latent) -> Video:
        logging.debug('VaeVideoEncode apply latent %s', latent)

        self.onload()
        latent.onload()

        v = self.decode(latent.latent)
        video = Video('HWC', video=v)

        logging.debug('VaeVideoEncode apply video %s', video)

        video.offload()
        latent.offload()
        self.offload()

        return video

    def decode(self, latents):
        video_length = latents.shape[2]
        latents = latents.to(device=self.vae.device, dtype=self.vae.dtype)
        latents = 1 / self.vae.config.scaling_factor * latents
        latents = rearrange(latents, "b c f h w -> (b f) c h w")

        kwargs = {}
        if isinstance(self.vae, AutoencoderKLTemporalDecoder):
            kwargs['num_frames'] = self.vae_batch

        video = []
        for frame_idx in tqdm(range(0, latents.shape[0], self.vae_batch)):
            frame = self.vae.decode(
                latents[frame_idx:frame_idx+self.vae_batch],
                **kwargs,
                ).sample
            video.append(frame.cpu())

        video = torch.cat(video).to('cuda')
        video = rearrange(video, "(b f) c h w -> b f h w c", f=video_length)
        video.div_(2).add_(0.5).clamp_(0, 1)
        video.mul_(255)

        return video.to(torch.uint8)
