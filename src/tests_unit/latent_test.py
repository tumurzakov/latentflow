import unittest
import torch

from latentflow.video import Video
from latentflow.latent import Latent
from latentflow.tile import Tile
from latentflow.prompt_embeddings import PromptEmbeddings

from video_encode_test import TestVideoEncode
from latent_decode_test import TestLatentDecode
from noise_test import TestNoise
from invert_test import TestInvert
from unet_test import TestUnet

class LatentTest(unittest.TestCase):

    def test_should_create_random_latent(self):
        latent = Latent(width=512//8, height=288//8, length=48)
        self.assertTrue(latent.latent is not None)

    def test_should_load_video(self):
        video = Video('HWC', torch.randn((48,288,512,3)))
        latent = TestVideoEncode().apply(video)
        self.assertTrue(isinstance(latent, Latent))

    def test_should_noise_latent(self):
        video = Video('HWC', torch.randn((48,288,512,3)))
        latent = TestVideoEncode().apply(video)
        latent = TestNoise().apply(latent)
        self.assertTrue(isinstance(latent, Latent))

    def test_should_invert_latent(self):
        video = Video('HWC', torch.randn((48,288,512,3)))
        latent = TestVideoEncode().apply(video)
        latent = TestInvert().apply(latent)
        self.assertTrue(isinstance(latent, Latent))

    def test_should_decode_latent(self):
        video = Video('HWC', torch.randn((48,288,512,3)))
        latent = TestVideoEncode().apply(video)
        video = TestLatentDecode().apply(latent)
        self.assertTrue(isinstance(video, Video))

    def test_should_substruct_noise(self):
        video = Video('HWC', torch.randn((48,288,512,3)))
        embeddings = PromptEmbeddings(embeddings=torch.randn((1,1,1)))
        latent = TestVideoEncode().apply(video)
        latent = TestNoise().apply(latent)
        timestep = torch.tensor(1)
        latent = TestUnet().apply(latent, embeddings, timestep)
        self.assertTrue(isinstance(latent, Latent))
