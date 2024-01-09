import torch

from latentflow.noise import Noise
from latentflow.latent import Latent

class TestNoise(Noise):
    def apply(self, latent: Latent) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))
