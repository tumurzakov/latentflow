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


from diffusers import AutoencoderKL
from diffusers.image_processor import VaeImageProcessor

from .latent_decode import LatentDecode
from .video import Video
from .latent import Latent

class VaeLatentDecode(LatentDecode):
    def __init__(self, vae: VideoVae = None, vae_batch=1):
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


class VideoVae:
    def __init__(self, config_path, weights_path):
        sd = load_safetensors(weights_path)
        prefix = 'first_stage_model.decoder.'
        weights = {}
        for key in sd.keys():
            if prefix in key:
                weights[key.replace(prefix, '')] = sd[key]
        del sd
        gc.collect()
        torch.cuda.empty_cache()

        config = OmegaConf.load(config_path)
        decoder = instantiate_from_config(config)
        m, e = decoder.load_state_dict(weights, strict=False)
        print("missing:", len(m), "expected:", len(e))
        self.decoder = decoder.eval().to('cuda', dtype=torch.float16)

    def decode(self, latents, n_samples=12):
        video_length = latents.shape[2]
        z = rearrange(latents, 'b c f h w -> (b f) c h w')
        n_rounds = math.ceil(z.shape[0] / n_samples)
        scale_factor = 0.18215
        z = 1.0 / scale_factor * z
        all_out = []
        with torch.autocast("cuda", dtype=torch.float16):
            for n in tqdm(range(n_rounds), desc='render'):
                timesteps = len(z[n * n_samples : (n + 1) * n_samples])
                out = self.decoder(z[n * n_samples : (n + 1) * n_samples], timesteps=timesteps)
                all_out.append(out)

        out = torch.cat(all_out, dim=0)
        out = rearrange(out, '(b f) c h w -> b c f h w', f=video_length)
        out = (out / 2 + 0.5).clamp(0, 1)

        out = out.detach().cpu().float()
        return out


