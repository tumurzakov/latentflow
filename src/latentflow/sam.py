import torch
import logging
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from einops import rearrange
from tqdm import tqdm
import numpy as np

from .flow import Flow
from .mask import Mask

class SegmentAnythingModel(Flow):
    def __init__(self, sam_model, onload_device='cuda', offload_device='cpu'):
        self.sam_model = sam_model
        self.onload_device = onload_device
        self.offload_device = offload_device

    def onload(self):
        self.sam_model.to(self.onload_device)

    def offload(self):
        self.sam_model.to(self.offload_device)

class LoadSegmentAnything(Flow):
    @classmethod
    def load(self, model_type, model_path):
        return SegmentAnythingModel(sam_model_registry[model_type](checkpoint=model_path))

class SegmentAnything(Flow):
    def __init__(self,
            sam_model,
            ):

        self.sam_model = sam_model

    def onload(self):
        self.sam_model.onload()

    def offload(self):
        self.sam_model.offload()

    def apply(self, video):

        self.onload()
        video.onload()

        generator = SamAutomaticMaskGenerator(self.sam_model.sam_model)

        batches = []
        for bi, b in enumerate(video.hwc()):
            frames = []
            for f in tqdm(b, desc=f'segment {bi}'):
                masks = generator.generate(f.detach().cpu().numpy().astype(np.uint8))
                masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
                masks = [torch.tensor(m['segmentation']).to(f.device) for m in masks[:1]]
                masks = torch.stack(masks).unsqueeze(0)
                frames.append(masks)
            frames = torch.stack(frames)
            batches.append(frames)

        batches = torch.stack(batches)

        batches = rearrange(batches, 'b f c m h w -> (b m) f h w c')
        batches = batches.repeat(1,1,1,1,3)

        result = Mask(batches, mode='b f h w c')

        result.offload()
        video.offload()
        self.offload()

        return result

