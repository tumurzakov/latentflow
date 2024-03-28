import torch
import logging
from diffusers import AutoencoderKL, AutoencoderKLTemporalDecoder

from .flow import Flow

class Vae(Flow):
    def __init__(self, vae, onload_device='cuda', offload_device='cpu'):
        self.vae = vae
        self.onload_device = onload_device
        self.offload_device = offload_device

    @classmethod
    def load(cls, model_path, is_svd=False, onload_device='cuda', offload_device='cpu'):
        if is_svd:
            vae = AutoencoderKLTemporalDecoder.from_pretrained(model_path, subfolder="vae")
        else:
            try:
                vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae")
            except:
                vae = AutoencoderKL.from_pretrained(model_path)

        return cls(vae)

    def onload(self):
        self.vae.to(self.onload_device)

    def offload(self):
        self.vae.to(self.offload_device)
