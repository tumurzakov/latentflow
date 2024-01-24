import torch
from PIL import Image
import logging
import numpy as np

from .flow import Flow
from .video import Video

class MandelbrotNoise(Flow):
    def __init__(self,
            shape,
            device='cuda',
            extent=(-3, -2.5, 2, 2.5),
            quality=100,
            ):

        self.shape = shape
        self.device = device
        self.extent = extent
        self.quality = quality

    def apply(self, other=None):

        batches = []
        for b in range(self.shape[0]):

            frames = []
            for f in range(self.shape[1]):

                channels = []
                for c in range(self.shape[2]):
                    channel = Image.effect_mandelbrot(self.shape[3:], self.extent, self.quality)
                    channels.append(torch.tensor(np.array(channel)))

                channels = torch.stack(channels)
                frames.append(channels)

            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)
        logging.debug("MandelbrotNoise apply %s %s %s", batches.shape, batches.device, batches.dtype)

        return Video('CHW', batches)


