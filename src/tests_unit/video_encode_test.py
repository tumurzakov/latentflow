import unittest
import torch

from latentflow.video import Video
from latentflow.latent import Latent
from latentflow.video_encode import VideoEncode

class TestVideoEncode(VideoEncode):
    def apply(self, video: Video) -> Latent:
        return Latent(latent=torch.randn((1,1,1,1,1)))
