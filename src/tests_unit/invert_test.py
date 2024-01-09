import torch
from latentflow.invert import Invert
from latentflow.latent import Latent

class TestInvert(Invert):
    def apply(self, latent: Latent) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))
