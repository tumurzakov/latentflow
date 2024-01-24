import os
import torch
from rembg import remove, new_session
import numpy as np

import logging
logging.basicConfig(level=logging.INFO)
from tqdm import tqdm

from .flow import Flow
from .video import Video

class VideoRembg(Flow):
    def __init__(self, model_name='u2net_human_seg', cache=None):
        self.cache = cache
        self.session = new_session(model_name)

    def apply(self, video):
        if self.cache is not None and os.path.isfile(self.cache):
            v = torch.load(self.cache).to(video.video.device)
        else:
            v = self.process(video)
            torch.save(v, self.cache)

        return Video('HWC', v)

    @torch.no_grad()
    def process(self, video):
        batches = []
        v = video.hwc()
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
