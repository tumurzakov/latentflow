import torch
import logging
import numpy as np
from tqdm import tqdm

from .flow import Flow
from .video import Video

from controlnet_aux.processor import Processor

class ControlNetProcessor(Flow):
    def __init__(self,
            processor_name,
            detect_resolution = 512,
            image_resolution = 512,
            onload_device = 'cuda',
            offload_device = 'cpu',
            ):
        self.processor_name = processor_name
        self.detect_resolution = detect_resolution
        self.image_resolution = image_resolution
        self.onload_device = onload_device
        self.offload_device = offload_device
        self.processor = None

    def onload(self):
        if self.processor is None:
            self.processor = Processor(self.processor_name, {
                'detect_resolution': self.detect_resolution,
                'image_resolution': self.image_resolution,
                })

            if hasattr(self.processor.processor, 'to'):
                self.processor.processor.to(self.onload_device)


    def offload(self):
        if hasattr(self.processor.processor, 'to'):
            self.processor.processor.to(self.offload_device)
        else:
            self.processor = None

    def apply(self, video):
        self.onload()
        video.onload()

        video_tensor = video.hwc()

        batches = []
        for i, v in enumerate(video_tensor):
            frames = []
            for f in tqdm(v, desc=f'{i+1}/{len(video_tensor)}'):
                img = self.processor(f.cpu(), to_pil=True)
                frames.append(torch.tensor(np.array(img)))
            frames = torch.stack(frames)
            batches.append(frames)
        batches = torch.stack(batches)
        result = Video('HWC', batches)

        result.offload()
        video.offload()
        self.offload()

        return result

