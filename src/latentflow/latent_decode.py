import torch
from .video import Video
from .latent import Latent

class LatentDecode:
    def apply(self, latent: Latent) -> Video:
        pass
