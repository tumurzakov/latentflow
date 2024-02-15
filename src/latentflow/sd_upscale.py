import os
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline
import torch
import numpy as np
from einops import rearrange
from tqdm import tqdm

import logging
from .flow import Flow
from .video import Video

class SDUpscale(Flow):

    def __init__(self, *args, **kwargs):
        self.pipeline = None
        self.kwargs = kwargs

        if 'prompt' not in self.kwargs:
            self.kwargs['prompt'] = "high quality, detailed"

    def apply(self, video: Video):
        upscaled = []
        for v in video.hwc():
            upscaled.append(self.upscale(v, **self.kwargs))

        upscaled = torch.stack(upscaled)

        return Video('HWC', upscaled)

    def load_pipeline(self):
        if self.pipeline is None:
            model_id = "stabilityai/stable-diffusion-x4-upscaler"
            self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
            self.pipeline = self.pipeline.to("cuda")
            self.pipeline.vae.enable_tiling = True
            self.pipeline.set_progress_bar_config(disable=True)

    def upscale(self,
            video,
            prompt,
            num_inference_steps=50,
            noise_level=20,
            video_length=48,
            width=512,
            height=288,
            guidance_scale=0.0,
            frame_limit=None,
            lora=None):

        if video_length is None:
            video_length = video.shape[0]

        self.load_pipeline()

        if lora is not None:
            self.pipeline.unet.load_attn_procs(lora)

        frames = []
        for i, frame in tqdm(enumerate(video), desc=f"frame {i}"):
            if frame_limit is not None and i >= frame_limit:
                break

            low_res_img = Image.fromarray(frame.numpy())

            upscaled_image = self.pipeline(
                    prompt=prompt,
                    image=low_res_img,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale,
                    noise_level=noise_level).images[0]

            frames.append(torch.tensor(np.array(upscaled_image)))

        frames = torch.stack(frames)

        return frames

