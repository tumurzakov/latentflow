import torch
import unittest

from latentflow.latent import Latent
from latentflow.prompt_embeddings import PromptEmbeddings
from latentflow.unet import Unet
from latentflow.diffuse import Diffuse
from latentflow.timesteps import Timesteps

from unet_test import TestUnet

class TestDiffuse:
    def apply(self,
            latent: Latent,
            embeddings: PromptEmbeddings,
            unet: Unet) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))

class DiffuseTest(unittest.TestCase):
    def test_should_diffuse(self):
        latent=Latent(latent=torch.randn((1,1,1,1,1)))
        embeddings = PromptEmbeddings(embeddings=torch.randn((1,1,1)))

        timestep = torch.tensor(1)

        unet = TestUnet()
        timesteps = Timesteps(torch.tensor([999.0]), 1)
        diffuse = Diffuse(timesteps=timesteps)
        latent = diffuse.apply({'latent': Latent(latent=torch.randn(1,1,1,1,1))})

        self.assertTrue(isinstance(latent, Latent))

