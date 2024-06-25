import os
import torch
import logging
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from .flow import Flow
from .video import Video

from IPython.display import display
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from einops import rearrange
import gc

class VideoFaceMask(Flow):
    def __init__(self,
            cache=None,
            padding_percent=0.05,
            resolution=512,
            box_limit=None,
            onload_device: str='cuda',
            offload_device: str='cpu',
            ):
        self.cache = cache
        self.resolution = resolution
        self.padding_percent = padding_percent
        self.box_limit = box_limit
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.mtcnn = MTCNN(image_size=self.resolution, device=self.onload_device)

    def offload(self):
        del self.mtcnn
        gc.collect()
        torch.cuda.empty_cache()
        self.mtcnn = None

    def apply(self, video):
        self.onload()
        video.onload()

        if self.cache is not None and os.path.isfile(self.cache):
            v = torch.load(self.cache).to(video.video.device)
        else:
            v = self.process(video)
            if self.cache is not None:
                torch.save(v, self.cache)

        result = Video('HWC', v)

        result.offload()
        video.offload()
        self.offload()
        return result

    @torch.no_grad()
    def process(self, video):
        batches = []
        v = video.hwc()
        for b in v:
            frames = []
            for f in tqdm(b, desc=f'facemask'):

                boxes, probs, points = self.mtcnn.detect(
                        f,
                        landmarks=True)

                back = torch.zeros_like(f)

                if boxes is not None and len(boxes) > 0:
                    for i,box in enumerate(boxes):

                        if self.box_limit is not None and i >= self.box_limit:
                            break

                        x0,y0,x1,y1 = [int(x) for x in box]
                        h = y1-y0
                        w = x1-x0
                        ph = int(h * self.padding_percent)
                        pw = int(w * self.padding_percent)
                        x0 = x0 - pw
                        x1 = x1 + pw
                        y0 = y0 - ph
                        y1 = y1 + ph

                        x0 = x0 if x0 > 0 else 0
                        y0 = y0 if y0 > 0 else 0

                        back[y0:y1,x0:x1,:] = 255

                frames.append(torch.tensor(back))

            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)
        batches = batches.to(v.device)

        return batches
