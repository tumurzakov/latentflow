import torch
import logging

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from tqdm import tqdm
from PIL import Image
import numpy as np
import gc

from .flow import Flow
from .video import Video

class RESRGANUpscale(Flow):
    def __init__(self, model_path, scale=4):
        self.model_path = model_path
        self.scale = scale

    def apply(self, other: Video) -> Video:

        other.onload()

        video = upscale(other.hwc(), self.model_path, scale=self.scale)
        video = Video('HWC', video)

        video.offload()
        other.offload()

        return video

def realesrgan(model_path, scale=4):
    if 'RealESRGAN_x4plus_anime_6B.pth' in model_path:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
    elif 'RealESRGAN_x2plus.pth' in model_path:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
    else:
        model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)

    upscaler = RealESRGANer(
        scale=scale,
        model_path=model_path,
        model=model)

    return upscaler

def upscale(video, model_path, scale=None, *args, **kwargs):

    sample = video.detach().cpu().numpy().astype(np.uint8)

    upscaler = realesrgan(model_path, *args, **kwargs)

    batches = []
    for b in range(sample.shape[0]):
        frames = []
        for f in tqdm(range(sample.shape[1]), desc='upscale'):
            frame = sample[b][f]
            upscaled = upscaler.enhance(frame, outscale=scale)[0]
            frames.append(torch.tensor(upscaled))

        frames = torch.stack(frames)
        batches.append(frames)

    batches = torch.stack(batches)

    return batches.to(video.device)
