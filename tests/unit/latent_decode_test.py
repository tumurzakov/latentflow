import torch

from latentflow.latent_decode import LatentDecode
from latentflow.video import Video
from latentflow.latent import Latent

class TestLatentDecode(LatentDecode):
    def apply(self, latent: Latent) -> Video:
        return Video('HWC', torch.randn((288,255,3)))
