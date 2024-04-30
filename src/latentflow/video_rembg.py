import os
import gc
import torch
from rembg import remove, new_session
import numpy as np
from einops import rearrange

import logging
from tqdm import tqdm

from .flow import Flow
from .video import Video

class VideoRembg(Flow):
    def __init__(self, model_name='u2net_human_seg', cache=None):
        self.cache = cache
        self.model_name = model_name
        self.session = None

    def onload(self):
        self.session = new_session(self.model_name)

    def offload(self):
        del self.session
        gc.collect()
        self.session = None

    def apply(self, video):
        self.onload()
        video.onload()

        if self.cache is not None and os.path.isfile(self.cache):
            v = torch.load(self.cache).to(video.video.device)
        else:
            v = self.process(video.hwc())
            if self.cache is not None:
                torch.save(v, self.cache)

        if len(v.shape) == 4:
            v = v.unsqueeze(0).repeat(3,1,1,1,1)
            v = rearrange(v, 'c b f h w -> b f h w c')

        shape = video.video.shape
        mate = v[
                :shape[0],
                :shape[1],
                :shape[2],
                :shape[3],
                :shape[4],
                ]

        output = Video('HWC', mate)
        logging.debug("VideoRembg %s %s", video, output)

        output.offload()
        self.offload()

        return output

    @torch.no_grad()
    def process(self, v):
        batches = []
        for b in v:
            frames = []
            for f in tqdm(b, desc=f'rembg'):
                output = remove(
                        f.detach().cpu().numpy(),
                        session=self.session,
                        only_mask=True,
                        post_process_mask=True)
                frames.append(torch.tensor(output))

            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)
        batches = batches.to(v.device)

        return batches
