import unittest
import torch

from latentflow.video import Video
from latentflow.latent import Latent
from latentflow.tile import Tile
from latentflow.prompt import Prompt

from prompt_encode_test import TestPromptEncode
from video_encode_test import TestVideoEncode
from noise_test import TestNoise
from unet_test import TestUnet

class PipeTest(unittest.TestCase):
    def test_should_infer(self):
        """
        video = Video('HWC', torch.randn((48,288,512,3)))
        latent = TestVideoEncode().apply(video)
        latent = TestNoise().apply(latent)

        prompt = Prompt()
        embeddings = TestPromptEncode().apply(prompt)

        scheduler = TestDiffuse()
        latent = scheduler.apply(
                latent,
                embeddings,
                TestUnet(),
                )

        self.assertTrue(isinstance(latent, Latent))
        """
