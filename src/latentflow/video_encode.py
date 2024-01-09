import torch

from .flow import Flow
from .video import Video
from .latent import Latent

class VideoEncode(Flow):
    def apply(self, video: Video) -> Latent:
        pass
