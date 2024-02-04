import torch
from .video import Video
from .latent import Latent
from .flow import Flow

class LatentDecode(Flow):
    def apply(self, latent: Latent) -> Video:
        pass
