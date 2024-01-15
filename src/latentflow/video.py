import torch
from einops import rearrange
from typing import List, Optional, Tuple, Union, Generator
from .flow import Flow

import logging

class Video(Flow):
    """
    :param mode: str, could be HWC or CWH
    :param video: torch.Tensor
    """
    def __init__(self,
            mode: str = 'HWC',
            video: torch.Tensor = None,
            device: Optional[Union[str, torch.device]] = None,
            ):
        self.video = video
        if mode == 'CHW':
            self.video = rearrange(self.video, 'b f c h w -> b f h w c')

        if device is not None:
            self.video = self.video

        logging.debug("Video init %s", self)

    def __str__(self):
        if self.video is not None:
            shape = self.video.shape
            device = self.video.device
            dtype = self.video.dtype
            return f'Video({shape}, {device}, {dtype})'

        return f'Video(None)'

    def size(self):
        return (self.video.shape[2], self.video.shape[3])

    def chw(self):
        return rearrange(self.video, 'b f h w c -> b f c h w')

    def hwc(self):
        return self.video

    def cnet(self):
        return self.chw().float()/255.0

