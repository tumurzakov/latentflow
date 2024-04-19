import os
import torch
from einops import rearrange
from typing import List, Optional, Tuple, Union, Generator
import torch.nn.functional as F
from tqdm import tqdm
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
            onload_device: Optional[Union[str, torch.device]] = 'cuda',
            offload_device: Optional[Union[str, torch.device]] = 'cpu',
            ):

        assert isinstance(mode, str), 'Mode must be first parameter of Video'

        self.video = video
        self.onload_device = onload_device
        self.offload_device = offload_device

        if mode == 'CHW':
            self.video = rearrange(self.video, 'b f c h w -> b f h w c')

    def onload(self):
        if self.video is not None:
            self.video = self.video.to(self.onload_device)

    def offload(self):
        if self.video is not None:
            self.video = self.video.to(self.offload_device)

    def __len__(self):
        return self.video.shape[1]

    def __str__(self):
        if self.video is not None:
            shape = self.video.shape
            device = self.video.device
            dtype = self.video.dtype
            return f'Video({shape}, {device}, {dtype})'

        return f'Video(None)'

    def __getitem__(self, key):
        return Video('HWC', self.video[key])

    def set(self, value):
        self.video[:] = value

    def size(self):
        return (self.video.shape[2], self.video.shape[3])

    def save(self, path, fps, start_frame=0):
        video = self.hwc()
        assert len(video.shape) == 5, "Must have 5 dims"

        if not isinstance(path, list):
            path = [path]

        assert len(path) == video.shape[0], "Path count must match batch size"

        for v, p in zip(video, path):
            if '%' in path:
                self.write_images(v, p, start_frame)
            elif p.endswith('.pth'):
                torch.save(v, p)
            else:
                self.write_video_cv2(v, p, fps)

        return self

    def scale(self, scale):
        v = self.chw().float()
        size = (int(v.shape[1]), int(v.shape[3]*scale), int(v.shape[4]*scale))
        return self.resize(size)

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

    def rescale(self, scale):
        input_size = self.size()
        output_size = (int(input_size[1] * scale), int(input_size[0] * scale))

        batches = []
        for b in self.hwc():
            frames = []
            for f in b:
                frame = cv2.resize(f.detach().cpu().numpy(), output_size, interpolation=cv2.INTER_LANCZOS4)
                frames.append(torch.tensor(frame))

            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)

        return Video('HWC', batches)


    def chw(self):
        return rearrange(self.video, 'b f h w c -> b f c h w')

    def hwc(self):
        return self.video

    def cnet(self):
        return Tensor(self.chw().float()/255.0)

    def write_images(self, frames, output_path, start_frame = 0):
        dirname = os.path.dirname(output_path)
        if not os.path.isdir(dirname):
            os.makedirs(dirname, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_number = start_frame + i
            img = Image.fromarray(frame.detach().cpu().numpy())
            img.save(output_path % frame_number)

    def write_video_cv2(self, frames, output_path, fps):

        if frames.dtype == torch.float32 and torch.std(frames) < 2.0:
            frames = (frames * 255).to(torch.uint8)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        f, h, w, c = frames.shape
        frame_size = (w, h)

        # Create the VideoWriter object
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Assuming you have a NumPy array of frames called 'frames'
        for frame in frames:
            frame = frame.detach().cpu().numpy()
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
