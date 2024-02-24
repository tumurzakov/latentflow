import torch

from latentflow.unet import Unet
from latentflow.latent import Latent
from latentflow.prompt_embeddings import PromptEmbeddings

class TestUnet:
    def apply(self, latent: Latent, embeddings: PromptEmbeddings, timestep: torch.Tensor) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))

