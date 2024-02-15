import torch
from einops import rearrange
from typing import List, Optional, Tuple, Union, Generator
import torch.nn.functional as F
from .flow import Flow
from .tensor import Tensor

import cv2
import numpy as np
import logging

from IPython.display import display
from PIL import Image

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

    def __str__(self):
        if self.video is not None:
            shape = self.video.shape
            device = self.video.device
            dtype = self.video.dtype
            return f'Video({shape}, {device}, {dtype})'

        return f'Video(None)'

    def __getitem__(self, key):
        return Video('HWC', self.video[key])

    def size(self):
        return (self.video.shape[2], self.video.shape[3])

    def save(self, path, fps):
        video = self.hwc()
        assert len(video.shape) == 5, "Must have 5 dims"

        if not isinstance(path, list):
            path = [path]

        assert len(path) == video.shape[0], "Path count must match batch size"

        for v, p in zip(video, path):
            self.write_video_cv2(v, p, fps)

        return self

    def resize(self, size):
        v = self.chw().float()
        v = rearrange(v, 'b f c h w -> b c f h w')
        v = F.interpolate(
                v,
                size=size,
                mode='trilinear',
                align_corners=False
                )
        v = rearrange(v, 'b c f h w -> b f c h w')
        return Video('CHW', video=v.to(torch.uint8))

    def chw(self):
        return rearrange(self.video, 'b f h w c -> b f c h w')

    def hwc(self):
        return self.video

    def cnet(self):
        return Tensor(self.chw().float()/255.0)

    def write_video_cv2(self, frames, output_path, fps):
        if isinstance(frames, torch.Tensor):
            frames = frames.detach().cpu().numpy()

        frames = np.array(frames)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        f, h, w, c = frames.shape
        frame_size = (w, h)

        # Create the VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Assuming you have a NumPy array of frames called 'frames'
        for frame in frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Ensure the frame is in uint8 format
            frame = np.uint8(frame)

            # Write the frame to the video file
            out.write(frame)

        # Release the VideoWriter object
        out.release()

class VideoAdd(Flow):
    def __init__(self, video, key=None, mask=None):
        self.video = video
        self.key = key
        self.mask = mask

    def apply(self, other):
        logging.debug("VideoAdd.apply %s[%s] %s %s", self.video, self.key, self.mask, other)

        s = self.video.video
        l = other.video

        if self.mask is None:
            m = torch.ones_like(l)
        else:
            m = self.mask.mask

        if self.key is not None:
            s[self.key] += l*m
        else:
            s[
                0:l.shape[0],
                0:l.shape[1],
                0:l.shape[2],
                0:l.shape[3],
                0:l.shape[4]
            ]+= l*m

        return self.video
