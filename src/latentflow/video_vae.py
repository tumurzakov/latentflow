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

class VideoVae:
    def __init__(self, config_path, weights_path, device, dtype):
        self.device = device
        self.dtype = dtype

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
        self.decoder = decoder.eval().to(device, dtype=dtype)

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
